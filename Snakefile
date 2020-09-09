
idealized_simulations = {"ideal_brain_3D_N100":"IdealizedBrain3DBrainSim"}
real_brain_simulations = {"MRIExampleSegmentation_Ncoarse":"MRISegmentedBrainSim",
                          "MRIExampleSegmentation_Nmid":"MRISegmentedBrainSim",
                          #"MRIExampleSegmentation_Nfine":"MRISegmentedBrainSim"
                          }


ruleorder: generateMeshFromStl > generateMeshFromCSG

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
        mem_mb=lambda wildcards, input: int(max(0.2e4*input.size_mb, 1000)),
        ncpus=lambda wildcards, input: 4 if input.size_mb < 1000 else 40
    shell:
        """
        singularity exec {params.sing_image} mpirun -n {resources.ncpus} \
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
        config = "config_files/{mesh}.yml"
    output:
        "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_labels.xdmf",
        "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_labels.h5",
        config = "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_config.yml",

    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec {params.sing_image} \
        python3 scripts/generateIdealizedBrainMesh.py \
        -c config_files/{wildcards.mesh}.yml -N {wildcards.resolution}
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

