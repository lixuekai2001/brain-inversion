
import numpy as np
idealized_simulations = {#"ideal_brain_3D_Ncoarse":"sinusBrainSim",
                         #"ideal_brain_3D_Ncoarse":"baladontBrainSim",
                         #"ideal_brain_3D_Nmid":"stdBrainSim"
                         }
real_brain_simulations = {"MRIExampleSegmentation_Nvcoarse":"sinusBrainSim",
                          "MRIExampleSegmentation_Nvcoarse":"baladontBrainSim",
                          "MRIExampleSegmentation_Ncoarse":"sinusBrainSim",
                          "MRIExampleSegmentation_Ncoarse":"baladontBrainSim",
                          "MRIExampleSegmentation_Nmid":"baladontBrainSim",
                        }
sing_bind = ""

env_params = {"singularity_bind": sing_bind}

movies = ["VentricularFlow", "SagittalPressure", "SagittalDisplacement"]

try:
    sing_bind = config["singularity_bind"]
except:
    pass

ruleorder: generateMeshFromStl > generateMeshFromCSG

# input size in mb, num cpus, ram in mb
ressource_from_inputsize = [(1, 4, 4000), (5, 4, 8000), (10, 12, 25000), (18, 12, 35000),
                            (25, 12, 40000),
                            (40, 20, 100000),(60, 32, 160000), (80, 40, 184000),
                            (100, 60, 300000)]


def estimate_ressources(wildcards, input, attempt):
    mem = 184000
    cpus = 40
    nodes = 1
    for res in ressource_from_inputsize:
        if res[0] > input.size_mb:
            cpus = int(min(res[1] + (attempt-1)*4, 40))
            mem = int(min(res[2]*1.3**(attempt -1), 184000))
            break
    nodes = int(np.ceil(cpus/40))
    tot_minutes = int(cpus*5 * 1.5**(attempt -1))
    mins = tot_minutes%60
    hours = tot_minutes//60
    return {"mem_mb":mem, "cpus":cpus, "nodes":nodes, "time":f"0{hours}:{mins}:00"}


rule all:
    input:
        expand("results/{sim}/movies/{movies}/{movies}.mp4", movies=movies,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations.items() ]),
        expand("results/{sim}/movies/{movies}/{movies}.mp4", movies=movies,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations.items() ]),
        expand("results/{sim}/flow_plots/ventr_CSF_flow.png",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations.items() ]),
        expand("results/{sim}/flow_plots/ventr_CSF_flow.png",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations.items() ]),
        expand("results/{sim}/pressure_plots/pressure_gradient_lateral_ventricles_top_sas.png",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations.items() ]),
        expand("results/{sim}/pressure_plots/pressure_gradient_lateral_ventricles_top_sas.png",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations.items() ]),

rule all_movies:
    input:
        expand("results/{sim}/movies/{movies}/{movies}.mp4", movies=movies,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations.items() ]),
        expand("results/{sim}/movies/{movies}/{movies}.mp4", movies=movies,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations.items() ]),

rule postP_all:
    input:
        expand("results/{sim}/plots/ventr_CSF_flow.png",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations.items() ]),
        expand("results/{sim}/plots/ventr_CSF_flow.png",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations.items() ]),
            

rule all_ideal:
    input:
        [f"results/{mesh}_{sim_name}/results.xdmf" for mesh, sim_name in idealized_simulations.items()]


rule all_real:
    input:
        [f"results/{mesh}_{sim_name}/results.xdmf" for mesh, sim_name in real_brain_simulations.items()]

rule all_meshes:
    input:
        [f"meshes/{mesh_name}/{mesh_name}.xdmf" for mesh_name in list(idealized_simulations.keys())],
        [f"meshes/{mesh_name}/{mesh_name}.xdmf" for mesh_name in list(real_brain_simulations.keys())]

rule ideal_meshes:
    input:
        [f"meshes/{mesh_name}/{mesh_name}.xdmf" for mesh_name in list(idealized_simulations.keys())]


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
        ntasks=lambda wildcards, input, attempt: estimate_ressources(wildcards, input, attempt)["cpus"],
        input_mem_mb=lambda wildcards, input, attempt: int(input.size_mb),
        time=lambda wildcards, input, attempt: estimate_ressources(wildcards, input, attempt)["time"],
    shell:
        """
        mpirun --bind-to core -n {resources.ntasks} /usr/bin/time -p \
        singularity exec \
        {sing_bind} \
        {params.sing_image} \
        python3 scripts/runFluidPorousBrainSim.py \
        -c {input.config} \
        -m {input.meshfile} \
        -o {output.outfile} && \
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
        /usr/bin/time -v \
        singularity exec \
        {sing_bind} \
        {params.sing_image} \
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
        /usr/bin/time -v \
        singularity exec \
        {sing_bind} \
        {params.sing_image} \
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
        /usr/bin/time -v \
        singularity exec \
        {sing_bind} \
        {params.sing_image} \
        python3 scripts/generateIdealizedBrainMesh.py \
        -c {input.config} -o {output.meshfile}
        """
   

rule makePressurePlots:
    input:
        sim_results="results/{mesh}_{sim_name}/results.xdmf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml"
    output:
        "results/{mesh}_{sim_name}/pressure_plots/pressure_gradient_lateral_ventricles_top_sas.png"
    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
         """
        singularity exec \
        {sing_bind} \
        {params.sing_image} \
        xvfb-run -a python3 scripts/pressurePlots.py \
        {wildcards.mesh} {wildcards.sim_name}
        """

rule makeFlowPlots:
    input:
        sim_results="results/{mesh}_{sim_name}/results.xdmf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml"
    output:
        "results/{mesh}_{sim_name}/flow_plots/cum_CSF_flow.png"
    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
         """
        singularity exec \
        {sing_bind} \
        {params.sing_image} \
        xvfb-run -a python3 scripts/flowPlots.py \
        {wildcards.mesh} {wildcards.sim_name}
        """


rule makeMovie:
    input:
        sim_results="results/{mesh}_{sim_name}/results.xdmf",
        subdomain_file="meshes/{mesh}/{mesh}.xdmf",
        sim_config_file="results/{mesh}_{sim_name}/config.yml",
    output:
        "results/{mesh}_{sim_name}/movies/{movie_name}/{movie_name}.mp4"
    params:
        sing_image="~/sing_images/biotstokes.simg"
    shell:
        """
        singularity exec \
        {sing_bind} \
        {params.sing_image} \
        xvfb-run -a python3 scripts/make{wildcards.movie_name}Movie.py \
        {wildcards.mesh} {wildcards.sim_name}
        """

