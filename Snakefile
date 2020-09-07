
simulations = {"ideal_brain_2D_N50":"IdealizedBrain2DBrainSim",
               "MRIExampleSegmentation_N15":"MRISegmentedBrainSim",
              }

rule all:
    input:
        [f"results/{mesh}_{sim_name}/results.xdmf" for mesh, sim_name in simulations.items()]


rule runBrainSim:
    input:
        "meshes/{mesh}/{mesh}.xdmf",
        "meshes/{mesh}/{mesh}_boundaries.xdmf",
        "config_files/{sim_name}.yml",
    output:
        "results/{mesh}_{sim_name}/results.xdmf"
    params:
        sing_image="~/sing_images/biotstokes.simg"
    log:
        "results/{mesh}_{sim_name}/log"
    shell:
        """
        singularity exec {params.sing_image} mpirun -n 2 \
        python3 scripts/runFluidPorousBrainSim.py \
        config_files/{wildcards.sim_name}.yml > {log}
        """

rule extractBoundaries:
    input:
        "meshes/{mesh}/{mesh}_labels.xdmf"
    output:
         "meshes/{mesh}/{mesh}_boundaries.xdmf",
         "meshes/{mesh}/{mesh}.xdmf"

    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec {params.sing_image} \
        python3 scripts/extractSubdomains.py \
        meshes/{mesh}_labels.xdmf
        """

rule generateMeshFromStl:
    input:
        "brainMeshBaseFiles/{mesh}/csf.stl",
        "brainMeshBaseFiles/{mesh}/parenchyma.stl",
         "config_files/{mesh}.yml"
    output:
        "meshes/{mesh}_N{resolution}/{mesh}_N{resolution}_labels.xdmf"
    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec {params.sing_image} \
        python3 scripts/VolFromStlSVMTK.py \
        config_files/{wildcards.mesh}.yml \
        meshes/{wildcards.mesh}_N{wildcards.resolution}/{wildcards.mesh}_N{wildcards.resolution}_labels.xdmf
        """

rule generateMeshFromCSG:
    input:
        "config_files/{mesh}.yml"
    output:
        "meshes/{mesh}.xdmf",
        "meshes/{mesh}_boundaries.xdmf"

    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec {params.sing_image} \
        python3 scripts/generateIdealizedBrainMesh.py \
        config_files/{mesh}.yml"
        """
   

