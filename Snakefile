
idealized_simulations = {"ideal_brain_3D_Ncoarse":"stdBrainSim",
                         "ideal_brain_3D_Nmid":"stdBrainSim"}
real_brain_simulations = {"MRIExampleSegmentation_Ncoarse":"stdBrainSim",
                          #"MRIExampleSegmentation_Nmid":"stdBrainSim",
                          #"MRIExampleSegmentation_Nfine":"stdBrainSim"
                          }


ruleorder: generateMeshFromStl > generateMeshFromCSG

# input size in mb, num cpus, ram in mb
ressource_from_inputsize = [(1, 4, 4000), (5, 4, 8000), (10, 8, 20000),(20, 12, 30000), (40, 12, 60000),
                            (80, 12, 80000), (120, 20, 120000), (180, 32, 140000), (240, 40, 184000) ]


def estimate_ressources(wildcards, input, attempt):
    mem = 184000
    cpus = 40
    for res in ressource_from_inputsize:
        if res[0] > input.size_mb:
            cpus = int(min(res[1]*1.3**(attempt -1), 40))
            mem = int(min(res[2]*1.3**(attempt -1), 184000))
            break
    
    return {"mem_mb":mem, "cpus":cpus}


rule all:
    input:
        [f"results/{mesh}_{sim_name}/results.xdmf" for mesh, sim_name in idealized_simulations.items()],
        [f"results/{mesh}_{sim_name}/results.xdmf" for mesh, sim_name in real_brain_simulations.items()]


rule all_idealized:
    input:
        [f"results/{mesh}_{sim_name}/results.xdmf" for mesh, sim_name in idealized_simulations.items()]


rule all_real:
    input:
        [f"results/{mesh}_{sim_name}/results.xdmf" for mesh, sim_name in real_brain_simulations.items()]

rule all_meshes:
    input:
        [f"meshes/{mesh_name}/{mesh_name}.xdmf" for mesh_name in list(idealized_simulations.keys()) + list(real_brain_simulations.keys())]


rule runBrainSim:
    input:
        "meshes/{mesh}/{mesh}.h5",
        "meshes/{mesh}/{mesh}_boundaries.xdmf",
        "meshes/{mesh}/{mesh}_boundaries.h5",
        config = "config_files/{sim_name}.yml",
        meshfile = "meshes/{mesh}/{mesh}.xdmf",
    output:
        outfile="results/{mesh}_{sim_name}/results.xdmf",
        config="results/{mesh}_{sim_name}/config.yml"
    params:
        sing_image="~/sing_images/biotstokes.simg"
    log:
        "results/{mesh}_{sim_name}/log"
    resources:
        mem_mb=lambda wildcards, input, attempt: estimate_ressources(wildcards, input, attempt)["mem_mb"],
        cpus=lambda wildcards, input, attempt: estimate_ressources(wildcards, input, attempt)["cpus"],
        input_mem_mb=lambda wildcards, input, attempt: int(input.size_mb)
    shell:
        """
        mpirun --bind-to core -n {resources.cpus} \
        singularity exec --bind $SCRATCH:/tmp/ {params.sing_image} \
        python3 scripts/runFluidPorousBrainSim.py \
        -c {input.config} \
        -m {input.meshfile} \
        -o {output.outfile} \
         > {log} && \
        cp {input.config} {output.config}
        """

rule extractBoundaries:
    input:
        "meshes/{mesh}/{mesh}_labels.xdmf",
        "meshes/{mesh}/{mesh}_labels.h5"
    output:
         "meshes/{mesh}/{mesh}_boundaries.xdmf",
         "meshes/{mesh}/{mesh}_boundaries.h5",
         "meshes/{mesh}/{mesh}.xdmf",
         "meshes/{mesh}/{mesh}.h5"
    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec {params.sing_image} \
        python3 scripts/extractSubdomains.py \
        meshes/{wildcards.mesh}/{wildcards.mesh}.xdmf
        """


rule generateMeshFromStl:
    input:
        "brainMeshBaseFiles/{mesh}/csf.stl",
        "brainMeshBaseFiles/{mesh}/parenchyma.stl",
        config = "config_files/{mesh}_N{resolution}.yml"
    output:
        "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_labels.h5",
        meshfile = "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_labels.xdmf",
        config = "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_config.yml",
    params:
        sing_image = "~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec {params.sing_image} \
        python3 scripts/VolFromStlSVMTK.py \
        {input.config} \
        {output.meshfile} && \
        cp {input.config} {output.config}
        """

rule generateMeshFromCSG:
    input:
        config = "config_files/{mesh}_N{resolution}.yml"
    output:
        "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_labels.h5",
        meshfile = "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_labels.xdmf",
        config = "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_config.yml",

    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec {params.sing_image} \
        python3 scripts/generateIdealizedBrainMesh.py \
        -c {input.config} -o {output.meshfile}
        """
   

rule postprocess:
    input:
        sim_results="results/{mesh}_{sim_name}/results.xdmf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml"
    output:
        "results/{mesh}_{sim_name}/plots/plot.pdf"
    log:
        notebook="results/{mesh}_{sim_name}/plots/postnb.ipynb"
    notebook:
        "notebooks/PostProcessBrainSim.ipynb"

