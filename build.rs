use anyhow::{Context, Result};
use minijinja::{context, value::Value, Environment};
use std::{
    fs::{self, File},
    io::{self, Write},
    path::Path,
};

fn compile_asm(name: &str, template_file: &str, context: &Value) -> Result<()> {
    let mut env = Environment::new();
    let out_dir = std::env::var("OUT_DIR")?;
    let filename = Path::new(template_file);
    println!("cargo:rerun-if-changed={}", filename.display());

    let template = fs::read_to_string(template_file)?;
    env.add_template(name, &template)?;

    let template = env.get_template(name)?;

    let dest_path = Path::new(&out_dir).join(format!("{}.s", name));
    let mut f = File::create(&dest_path)?;
    template.render_to_write(context, &mut f)?;

    println!("Wrote {}.", dest_path.display());

    cc::Build::new()
        .file(&dest_path)
        .compile(&format!("{}", name));

    println!("Compiled {}.", dest_path.display());

    Ok(())
}

#[derive(Debug)]
struct GemmKernel {
    template: &'static str,
    name: String,
    contig_dim: usize,
    tile_dim: usize,
    a_ty: &'static str,
    b_ty: &'static str,
    c_ty: &'static str,
    align_contig_to: usize,
    loop_order: &'static str, 
}

impl GemmKernel {
    fn fmt_extern(&self, suffix: &str, mut f: impl Write) -> io::Result<()> {
        let name = &self.name;
        write!(f, "    #[link_name = \"{name}_{suffix}\"]\n")?;
        write!(
            f,
            "    fn {name}(info: *const TileKernelArguments) -> isize;\n"
        )
    }

    fn fmt_def(&self, mut f: impl Write) -> io::Result<()> {
        let Self {
            name,
            contig_dim,
            tile_dim,
            a_ty: _,
            b_ty: _,
            c_ty: _,
            align_contig_to: _,
            loop_order,
            ..
        } = self;

        write!(f, "#[allow(non_camel_case_types)]\n")?;
        write!(f, "pub const {}: AVX2Kernel = AVX2Kernel {{\n", name.to_uppercase())?;
        write!(f, "    func: {name},\n")?;
        write!(f, "    r: {contig_dim},\n")?;
        write!(f, "    s: {tile_dim},\n")?;
        write!(f, "    loop_order: LoopOrder::{},\n", loop_order.to_uppercase())?;
        write!(f, "}};\n")
    }

    fn compile(&self, suffix: &str) -> Result<()> {
        compile_asm(
            &self.name,
            self.template,
            &context! {
                SUFFIX => suffix,
                DIM_R => self.contig_dim,
                DIM_S => self.tile_dim,
                LOOP_ORDER => self.loop_order,
            },
        )
    }
}

fn add_avx2_kernels(kernels: &mut Vec<GemmKernel>) {
    let template = "./asm/fma_sss.s";

    for loop_order in ["acb", "cab", "abc"] {
        for dim1 in 1..16 {
            for dim2 in 1..16 {
                let nb_reg = dim1 * dim2 + dim1 + 1;
                if nb_reg <= 16 {
                    kernels.push(GemmKernel {
                        template,
                        name: format!("mmkernel_avx2_sss_{}x{dim2}_{loop_order}", dim1 * 8),
                        contig_dim: dim1 * 8,
                        tile_dim: dim2,
                        a_ty: "f32",
                        b_ty: "f32",
                        c_ty: "f32",
                        align_contig_to: 32,
                        loop_order,
                    });
                }
            }
        }
    }
}

fn main() {
    let mut kernels = Default::default();
    add_avx2_kernels(&mut kernels);

    let suffix = env!("CARGO_PKG_VERSION")
        .replace('-', "_")
        .replace('.', "_");

    // compile_asm(
    //     "avx2_inner",
    //     "./asm/avx2_inner.s",
    //     &suffix,
    //     &context! { suffix },
    // ).context("compiling avx2_inner.s").unwrap();

    for k in &kernels {
        k.compile(&suffix)
            .with_context(|| format!("compiling kernel {k:?}"))
            .unwrap();
    }

    make_extern_kernels_file(&suffix, &kernels)
        .context("writing extern kernels rust file")
        .unwrap();
}

fn make_extern_kernels_file(suffix: &str, kernels: &[GemmKernel]) -> Result<()> {
    let binding = std::env::var("OUT_DIR")?;
    let out_dir = Path::new(&binding);
    let mut f = File::create(out_dir.join("extern_kernels.rs"))?;
    write!(
        f,
        "// This file is autogenerated by the build.rs file. Do not touch!\n"
    )?;
    write!(f, "use crate::kernel::{{AVX2Kernel, LoopOrder, TileKernelArguments}};\n")?;
    write!(f, "\nextern \"C\" {{\n")?;
    for kernel in kernels {
        kernel.fmt_extern(suffix, &mut f)?;
    }
    write!(f, "}}\n")?;
    write!(f, "\n")?;
    for kernel in kernels {
        kernel.fmt_def(&mut f)?;
    }
    write!(f, "pub const KERNELS: &[AVX2Kernel] = &[\n")?;
    for kernel in kernels {
        write!(f, "    {},\n", kernel.name.to_uppercase())?;
    }
    write!(f, "];\n")?;

    Ok(())
}
