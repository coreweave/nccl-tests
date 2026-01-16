#!/usr/bin/env nu

# NCCL Test CLI - Sync cluster GPU types and generate manifests

# Get default image from workflow file and git SHA
def get-default-image [] {
    let workflow = (open .github/workflows/ubuntu-22.yml)
    let job = $workflow.jobs.cu129.with
    let sha = (git rev-parse --short HEAD)
    $"ghcr.io/coreweave/nccl-tests:($job.base-tag)-nccl($job.nccl-version)-($sha)"
}

# Get image tag arguments for CUE commands
def get-image-tags [] {
    ["-t", $"defaultImage=(get-default-image)"]
}

# Generate CUE config file content from node data using cue eval
def generate_cue_file [node: record] {
    let labels = $node.metadata.labels
    let cap = $node.status.capacity

    let gpu_count = $labels | get "gpu.nvidia.com/count" | into int
    let node_type = $labels | get "node.coreweave.cloud/type"
    let cpu = $labels | get "cpu.coreweave.cloud/cores" | into int
    let mem_ki = $cap.memory | str replace "Ki" "" | into int
    let mem_gi = ($mem_ki / 1048576 | math floor)
    let ib_speed = $labels | get -o "ib.coreweave.cloud/speed" | default "0G"
    let has_ib = $ib_speed != "0G" and $ib_speed != ""
    let config_name = $node_type | str replace "-" "_" --all
    # Detect IB devices from node labels (pattern: ib.coreweave.cloud/neighbors.current.ibp<N>.device)
    let ib_device_labels = ($labels | columns | where { |k| $k =~ 'ib.coreweave.cloud/neighbors.current.ibp\d+\.device' })
    let ib_devices = if ($ib_device_labels | length) > 0 {
        $ib_device_labels
        | each { |k| $k | parse --regex 'ibp(?P<num>\d+)' | get num.0 | into int }
        | sort
        | each { |n| $"ibp($n):1" }
        | str join ","
    } else {
        ""
    }

    let img_tags = (get-image-tags)
    let content = (cue eval ./utils/... -e newGpu
        ...$img_tags
        -t $"nodeType=($node_type)"
        -t $"gpuCount=($gpu_count)"
        -t $"cpu=($cpu)"
        -t $"memGi=($mem_gi)"
        -t $"hasIB=($has_ib)"
        -t $"ibDevices=($ib_devices)")

    {
        name: $config_name
        content: $"package gpus\n\n($content)\n"
    }
}

# Sync cluster GPU types with CUE config
def "main sync" [
    --kubeconfig: string = "",  # Path to kubeconfig file
] {
    let kubeconfig = if $kubeconfig != "" { $kubeconfig | path expand } else { "" }
    let kc = if $kubeconfig != "" { ["--kubeconfig", $kubeconfig] } else { [] }

    # Get GPU types from cluster
    print -e "Fetching GPU nodes from cluster..."
    let cluster_nodes = (kubectl ...$kc get nodes -l node.coreweave.cloud/class=gpu -o json
        | from json
        | get items)

    let cluster_gpus = ($cluster_nodes
        | each { $in.metadata.labels | get "node.coreweave.cloud/type" }
        | uniq)

    # Get GPU configs from CUE
    print -e "Reading GPU configs from CUE..."
    let img_tags = (get-image-tags)
    let cue_gpus_raw = (cue export ./gpus/... -e gpus ...$img_tags | from json)
    let cue_gpu_names = ($cue_gpus_raw | columns)

    let cue_gpus = ($cue_gpu_names | each { |name|
        let cfg = ($cue_gpus_raw | get $name)
        {name: $name, nodeType: $cfg.nodeType}
    })

    let cue_node_types = ($cue_gpus | get nodeType)

    # Build status table
    let all_types = ($cluster_gpus | append $cue_node_types | uniq | sort)

    let status = ($all_types | each { |t|
        let in_cluster = $t in $cluster_gpus
        let in_config = $t in $cue_node_types
        let config_name = if $in_config {
            ($cue_gpus | where nodeType == $t | get name.0)
        } else { "" }

        {
            "GPU Type": $t
            "Cluster": (if $in_cluster { "✓" } else { "✗" })
            "Config": (if $in_config { $"✓ \(($config_name)\)" } else { "✗" })
        }
    })

    print ""
    print ($status | table)

    # Find missing configs
    let missing = ($cluster_gpus | where { |t| $t not-in $cue_node_types })

    if ($missing | length) > 0 {
        print $"\n($missing | length) GPU type\(s\) in cluster without config:\n"

        for gpu_type in $missing {
            let create = (input $"Create config for ($gpu_type)? [Y/n] ")
            if $create == "" or $create == "y" or $create == "Y" {
                # Get node details
                let node = (kubectl ...$kc get nodes -l $"node.coreweave.cloud/type=($gpu_type)" -o json
                    | from json
                    | get items.0)

                let cue_file = (generate_cue_file $node)
                let file_path = $"gpus/($cue_file.name).cue"
                $cue_file.content | save -f $file_path
                print $"Created ($file_path)"
            }
        }
    } else {
        print "\nAll cluster GPU types have configs ✓"
    }
}

# Generate NCCL test manifest
def "main generate" [
    --kubeconfig: string = "",  # Path to kubeconfig file
    --gpu: string = "",         # GPU type (skip prompt if provided)
    --scale: int = 0,           # Number of GPUs (skip prompt if provided)
    --multirack,                # Enable multi-rack topology spread
    --noib,                     # Disable InfiniBand
    --gdrcopy,                  # Enable GDRCopy
    --sharp,                    # Enable SHARP/COLLNET
    --apply,                    # Apply manifest to cluster
    --image: string = "",       # Override container image
] {
    let kubeconfig = if $kubeconfig != "" { $kubeconfig | path expand } else { "" }

    # Get image tags from workflow
    let img_tags = (get-image-tags)

    # Get available GPUs from CUE
    let gpus_json = (cue export ./gpus/... -e gpus ...$img_tags | from json)
    let gpu_names = ($gpus_json | columns)

    # Select GPU
    let selected_gpu = if $gpu != "" { $gpu } else {
        $gpu_names | input list "Select GPU type:"
    }

    if $selected_gpu not-in $gpu_names {
        print -e $"Error: GPU '($selected_gpu)' not found. Available: ($gpu_names | str join ', ')"
        exit 1
    }

    let gpu_info = ($gpus_json | get $selected_gpu)
    let gpus_per_node = $gpu_info.gpusPerNode

    # Select scale
    let selected_scale = if $scale > 0 { $scale } else {
        let default = $gpus_per_node * 8
        let scale_input = (input $"Number of GPUs [default: ($default), ($gpus_per_node)/node]: ")
        if $scale_input == "" { $default } else { $scale_input | into int }
    }

    # Validate scale is divisible by gpusPerNode
    if ($selected_scale mod $gpus_per_node) != 0 {
        print -e $"Error: Scale ($selected_scale) must be divisible by gpusPerNode ($gpus_per_node)"
        exit 1
    }

    let workers = ($selected_scale / $gpus_per_node | into int)
    print -e $"Generating: ($selected_gpu) × ($selected_scale) GPUs \(($workers) workers\)\n"

    # Generate manifest using CUE
    let mr = if $multirack { ["-t", "multirack=true"] } else { [] }
    let noib_flag = if $noib { ["-t", "noib=true"] } else { [] }
    let gdrcopy_flag = if $gdrcopy { ["-t", "gdrcopy=true"] } else { [] }
    let sharp_flag = if $sharp { ["-t", "sharp=true"] } else { [] }
    let img_override = if $image != "" { ["-t", $"image=($image)"] } else { [] }
    let manifest = (cue export -e manifest --out yaml -t $"gpu=($selected_gpu)" -t $"scale=($selected_scale)" ...$mr ...$noib_flag ...$gdrcopy_flag ...$sharp_flag ...$img_tags ...$img_override .)

    if $apply {
        let kc = if $kubeconfig != "" { ["--kubeconfig", $kubeconfig] } else { [] }
        $manifest | kubectl ...$kc apply -f -
    } else {
        print $manifest
    }
}

def main [] {
    print "NCCL Test CLI

Commands:
  sync      - Compare cluster GPU types vs CUE configs, create missing
  generate  - Interactive manifest generation

Options:
  --kubeconfig <path>  - Path to kubeconfig file
  --noib               - Disable InfiniBand
  --gdrcopy            - Enable GDRCopy
  --sharp              - Enable SHARP/COLLNET

Examples:
  ./nccl.nu sync --kubeconfig ~/Downloads/kubeconfig
  ./nccl.nu generate
  ./nccl.nu generate --gpu a100 --scale 64
  ./nccl.nu generate --gpu a100 --scale 64 --noib
  ./nccl.nu generate --gpu a100 --scale 64 --gdrcopy
  ./nccl.nu generate --gpu a100 --scale 64 --sharp
  ./nccl.nu generate --gpu h100 --scale 64 --sharp
  ./nccl.nu generate --gpu h100 --scale 128 --apply --kubeconfig ~/kube.conf
  ./nccl.nu generate --gpu h100 --scale 64 --image ghcr.io/coreweave/nccl-tests:custom-tag"
}
