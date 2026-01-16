package main

import (
	"strings"

	corev1 "cue.dev/x/k8s.io/api/core/v1"
	metav1 "cue.dev/x/k8s.io/apimachinery/pkg/apis/meta/v1"
	"github.com/coreweave/nccl-tests/gpus"
)

// Input parameters (set via -t flag)
gpu:          *"a100" | string @tag(gpu)
scale:        *64 | int        @tag(scale,type=int)
name:         *"" | string     @tag(name)
multirack:    *false | bool    @tag(multirack,type=bool)
noib:         *false | bool    @tag(noib,type=bool)
gdrcopy:      *false | bool    @tag(gdrcopy,type=bool)
sharp:        *false | bool    @tag(sharp,type=bool)
image:        *"" | string     @tag(image)
defaultImage: string           @tag(defaultImage)

// Lookup the GPU config
_gpu: gpus.gpus[gpu]

// Calculate workers (integer division)
_workers: scale div _gpu.gpusPerNode

// Resolve image (override takes precedence, then GPU-specific, then default)
_image: string
if image != "" {
	_image: image
}
if image == "" {
	_image: defaultImage
}

// Generate job name (lowercase)
_name: string
if name != "" {
	_name: name
}
if name == "" {
	_name: "nccl-test-\(scale)-\(strings.ToLower(_gpu.name))"
}

// Build env var flags string from mpiEnv list
_gpuEnvFlags: strings.Join([for k, v in _gpu.mpiEnv {"-x \(k)=\(v)"}], " \\\n  ")

// Build gdrcopy env flags
_gdrcopyEnvFlags: string
if gdrcopy {
	_gdrcopyEnvFlags: "-x NCCL_GDRCOPY_ENABLE=1 \\\n  -x NCCL_DEBUG=INFO \\\n  -x NCCL_DEBUG_SUBSYS=INIT"
}
if !gdrcopy {
	_gdrcopyEnvFlags: ""
}

// Build sharp env flags (COLLNET + ALGO for H100)
_sharpEnvFlags: string
if sharp && gpu == "h100" {
	_sharpEnvFlags: "-x NCCL_COLLNET_ENABLE=1 \\\n  -x NCCL_ALGO=COLLNETCHAIN"
}
if sharp && gpu != "h100" {
	_sharpEnvFlags: "-x NCCL_COLLNET_ENABLE=1"
}
if !sharp {
	_sharpEnvFlags: ""
}

// Combine all extra env flags
_allExtraEnv: [for s in [_gpuEnvFlags, _gdrcopyEnvFlags, _sharpEnvFlags] if s != "" {s}]
_extraEnvStr: strings.Join(_allExtraEnv, " \\\n  ")
_extraEnv:    string
if _extraEnvStr != "" {
	_extraEnv: " \\\n  \(_extraEnvStr)"
}
if _extraEnvStr == "" {
	_extraEnv: ""
}

// Determine IB HCA setting (eth0 when noib, else ibp)
_ibHca: string
if noib {
	_ibHca: "eth0"
}
if !noib {
	_ibHca: "ibp"
}

// Build UCX_NET_DEVICES part (only when has ibDevices and not noib and not gdrcopy)
_ucxPart: string
if _gpu.ibDevices != "" && !noib && !gdrcopy {
	_ucxPart: " \\\n  -x UCX_NET_DEVICES=\(_gpu.ibDevices)"
}
if _gpu.ibDevices == "" || noib || gdrcopy {
	_ucxPart: ""
}

// Build bind-to part (only when not gdrcopy)
_bindPart: string
if !gdrcopy {
	_bindPart: " \\\n  -bind-to none"
}
if gdrcopy {
	_bindPart: ""
}

// Build the mpirun command
_mpirunArgs: "mpirun \\\n  -np \(scale)\(_bindPart) \\\n  -x LD_LIBRARY_PATH \\\n  -x NCCL_SOCKET_IFNAME=eth0 \\\n  -x NCCL_IB_HCA=\(_ibHca)\(_ucxPart)\(_extraEnv) \\\n  /opt/nccl_tests/build/all_reduce_perf -b 512M -e 8G -f 2 -g 1\n"

// Launcher container spec (typed)
_launcherContainer: corev1.#Container & {
	name:  "nccl"
	image: _image
	env: [
		{name: "OMPI_ALLOW_RUN_AS_ROOT", value: "1"},
		{name: "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM", value: "1"},
	]
	command: ["/bin/bash", "-c"]
	args: [_mpirunArgs]
	resources: requests: {cpu: "2", memory: "128Mi"}
}

// Determine if we should request IB resources (GPU supports IB and noib is not set)
_useIB: _gpu.useIB && !noib

// Worker volume mounts (always dshm, optionally gdrdrv)
_workerVolumeMounts: [...corev1.#VolumeMount]
if !gdrcopy {
	_workerVolumeMounts: [{mountPath: "/dev/shm", name: "dshm"}]
}
if gdrcopy {
	_workerVolumeMounts: [
		{mountPath: "/dev/shm", name: "dshm"},
		{mountPath: "/dev/gdrdrv", name: "gdrdrv"},
	]
}

// Worker container spec (typed)
_workerContainer: corev1.#Container & {
	name:  "nccl"
	image: _image
	resources: {
		requests: {
			cpu:              "\(_gpu.cpu)"
			memory:           _gpu.memory
			"nvidia.com/gpu": "\(_gpu.gpusPerNode)"
			if _useIB {"rdma/ib": "1"}
		}
		limits: {
			"nvidia.com/gpu": "\(_gpu.gpusPerNode)"
			if _useIB {"rdma/ib": "1"}
			memory: _gpu.memory
			cpu:    "\(_gpu.cpu)"
		}
	}
	volumeMounts: _workerVolumeMounts
}

// Worker pod affinity (typed)
_workerAffinity: corev1.#Affinity & {
	if _gpu.podAffinity {
		podAffinity: preferredDuringSchedulingIgnoredDuringExecution: [{
			weight: 100
			podAffinityTerm: {
				labelSelector: matchLabels: "metadata.coreweave.cloud/job": "nccl-test"
				topologyKey: "topology.kubernetes.io/zone"
			}
		}]
	}
	nodeAffinity: requiredDuringSchedulingIgnoredDuringExecution: nodeSelectorTerms: [{
		matchExpressions: [{
			key:      _gpu.nodeKey
			operator: "In"
			values: [_gpu.nodeType]
		}]
	}]
}

// Worker volumes (always dshm, optionally gdrdrv)
_workerVolumes: [...corev1.#Volume]
if !gdrcopy {
	_workerVolumes: [{name: "dshm", emptyDir: medium: "Memory"}]
}
if gdrcopy {
	_workerVolumes: [
		{name: "dshm", emptyDir: medium: "Memory"},
		{name: "gdrdrv", hostPath: path: "/dev/gdrdrv"},
	]
}

// Topology spread constraints (typed)
_topologySpreadConstraints: [...corev1.#TopologySpreadConstraint]
if multirack {
	_topologySpreadConstraints: [{
		maxSkew:           1
		minDomains:        2
		topologyKey:       "topology.kubernetes.io/zone"
		whenUnsatisfiable: "DoNotSchedule"
		labelSelector: matchLabels: "metadata.coreweave.cloud/job": "nccl-test"
	}]
}

// The MPIJob manifest
manifest: {
	apiVersion: "kubeflow.org/v2beta1"
	kind:       "MPIJob"
	metadata: metav1.#ObjectMeta & {
		name: _name
	}
	spec: {
		slotsPerWorker: _gpu.gpusPerNode
		runPolicy: cleanPodPolicy: "Running"
		mpiReplicaSpecs: {
			Launcher: {
				replicas: 1
				template: {
					spec: corev1.#PodSpec & {
						containers: [_launcherContainer]
					}
				}
			}
			Worker: {
				replicas: _workers
				template: {
					metadata: metav1.#ObjectMeta & {
						labels: "metadata.coreweave.cloud/job": "nccl-test"
					}
					spec: corev1.#PodSpec & {
						topologySpreadConstraints: _topologySpreadConstraints
						containers: [_workerContainer]
						affinity: _workerAffinity
						volumes:  _workerVolumes
					}
				}
			}
		}
	}
}
