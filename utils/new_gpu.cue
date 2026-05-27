package utils

import (
	"strings"
	"github.com/coreweave/nccl-tests/schema"
)

// Input parameters via tags
_nodeType:     string @tag(nodeType)
_gpuCount:     int    @tag(gpuCount,type=int)
_cpu:          int    @tag(cpu,type=int)
_memGi:        int    @tag(memGi,type=int)
_hasIB:        bool   @tag(hasIB,type=bool)
_ibDevices:    string @tag(ibDevices)
_defaultImage: string @tag(defaultImage)

_configName: strings.Replace(_nodeType, "-", "_", -1)

// The generated config - validated against #GPU schema
_newConfig: schema.#GPU & {
	name:        strings.ToUpper(_nodeType)
	image:       _defaultImage
	gpusPerNode: _gpuCount
	nodeType:    _nodeType
	cpu:         _cpu
	memory:      "\(_memGi)Gi"
	if _hasIB {
		ibDevices: _ibDevices
	}
	if !_hasIB {
		useIB: false
	}
}

// Output structure for file generation
newGpu: gpus: (_configName): _newConfig
