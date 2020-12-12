
import numpy as np
import yaml
idealized_simulations = [#"ideal_brain_3D_Ncoarse":"sinusBrainSim",
                         #"ideal_brain_3D_Ncoarse":"baladontBrainSim",
                         #"ideal_brain_3D_Nmid":"stdBrainSim"
                        ]

real_brain_simulations = [#"MRIExampleSegmentation_Nvcoarse":"sinusBrainSim",
                          #("MRIExampleSegmentation_Ncoarse","standard"),
                          ("MRIExampleSegmentation_Nvcoarse","standard"),
                          ("MRIExampleSegmentation_Ncoarse","standard"),
                          #("MRIExampleSegmentation_Nmid","standard"),
                          #("MRIExampleSegmentation_Nvcoarse","outflowresistance"),
                          #("MRIExampleSegmentation_Nvcoarse","qvarlander"),
                          #("MRIExampleSegmentation_Nvcoarse","steadyStokes"),
                          #("MRIExampleSegmentation_NreducedCSF","standard"),
                          #("MRIExampleSegmentation_NreducedCSF","steadyStokes"),
                          #("MRIExampleSegmentation_NvreducedCSF","standard"),
                          #("MRIExampleSegmentation_NthinAQ","standard"),
                          ("MRIExampleSegmentation_Nvcoarse","ModelA"),
                          ("MRIExampleSegmentation_Nvcoarse","ModelB"),
                          ("MRIExampleSegmentation_Nvcoarse","ModelC"),
                          ("MRIExampleSegmentation_Nvcoarse","ModelD"),
                        ]
sing_bind = "--bind /run/media/marius/TOSHIBA\ EXT/results/:/run/media/marius/TOSHIBA\ EXT/results/"

env_params = {"singularity_bind": sing_bind}

movies = ["PressureFlow", "SagittalPressure", "SagittalDisplacement", "FluidVelocity"] #"VentricularFlow"]

try:
    sing_bind = config["singularity_bind"]
except:
    pass

ruleorder: generateMeshFromStl > generateMeshFromCSG

# input size in mb, num cpus, ram in mb
resource_from_inputsize = [(1, 4, 4000), (5, 8, 12000), (10, 8, 12000), (18, 12, 35000),
                            (25, 12, 40000),
                            (40, 20, 100000),(60, 24, 100000), (80, 40, 184000),
                            (100, 60, 300000)]


def estimate_resources(wildcards, input, attempt):
    mem = 184000
    cpus = 40
    nodes = 1
    with open(input.config,"r") as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)
    with open("config_files/saga_resources.yml","r") as config_file:
        resources = yaml.load(config_file, Loader=yaml.FullLoader)

    resources = resources[wildcards.mesh]
    cpus = resources["cpus"]
    mem = resources["mem"]
    total_secs = resources["secs_for_factorization"] + resources["secs_per_solve"]*config["num_steps"]

    tot_minutes = total_secs//60
    mins = tot_minutes%60
    hours = tot_minutes//60
    
    #for res in resource_from_inputsize:
    #    if res[0] >= input.size_mb:
    #        cpus = int(min(res[1] + (attempt-1)*4, 40))
    #        mem = int(min(res[2]*1.3**(attempt -1), 184000))
    #        break
    #nodes = int(np.ceil(cpus/40))
    #tot_minutes = int(cpus*6 * 1.5**(attempt -1))
    #mins = tot_minutes%60
    #hours = tot_minutes//60

    return {"mem_mb":mem, "cpus":cpus, "nodes":nodes, "time":f"0{hours}:{mins}:00"}


rule all:
    input:
        #expand("results/{sim}/movies/{movies}/{movies}.mp4", movies=movies,
        #        sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations ]),
        expand("results/{sim}/movies/{movies}/{movies}_array_plot.pdf", movies=movies,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),
        expand("results/{sim}/flow_key_quantities.yml",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations ]),
        expand("results/{sim}/flow_key_quantities.yml",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),
        expand("results/{sim}/pressure_key_quantities.yml",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations ]),
        expand("results/{sim}/pressure_key_quantities.yml",
                    sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),

rule all_movies:
    input:
        expand("results/{sim}/movies/{movies}/{movies}.mp4", movies=movies,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in idealized_simulations ]),
        expand("results/{sim}/movies/{movies}/{movies}_array_plot.pdf", movies=movies,
                sim=[f"{mesh}_{sim_name}" for mesh, sim_name in real_brain_simulations ]),



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
        mem_mb=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["mem_mb"],
        ntasks=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["cpus"],
        input_mem_mb=lambda wildcards, input, attempt: int(input.size_mb),
        time=lambda wildcards, input, attempt: estimate_resources(wildcards, input, attempt)["time"],
    shell:
        """
        srun --cpu-bind=verbose -m block:block -n {resources.ntasks} \
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
        "results/{mesh}_{sim_name}/pressure_key_quantities.yml"
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
        "results/{mesh}_{sim_name}/flow_key_quantities.yml"
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
        "results/{mesh}_{sim_name}/movies/{movie_name}/{movie_name}_array_plot.pdf"
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

