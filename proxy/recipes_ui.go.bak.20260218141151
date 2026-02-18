package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"io/fs"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/mostlygeek/llama-swap/proxy/config"
	"gopkg.in/yaml.v3"
)

const (
	recipesBackendDirEnv              = "LLAMA_SWAP_RECIPES_BACKEND_DIR"
	recipesBackendOverrideFileEnv     = "LLAMA_SWAP_RECIPES_BACKEND_OVERRIDE_FILE"
	recipesLocalDirEnv                = "LLAMA_SWAP_LOCAL_RECIPES_DIR"
	trtllmSourceImageOverrideFile     = ".llama-swap-trtllm-source-image"
	nvidiaSourceImageOverrideFile     = ".llama-swap-nvidia-source-image"
	defaultRecipesBackendSubdir       = "spark-vllm-docker"
	defaultRecipesBackendAltSubdir    = "spark-trtllm-docker"
	defaultRecipesBackendSQLSubdir    = "spark-sqlang-docker"
	defaultRecipesBackendNvidiaSubdir = "spark-vllm-docker-nvidia"
	defaultRecipesLocalSubdir         = "llama-swap/recipes"
	defaultRecipeGroupName            = "managed-recipes"
	defaultTRTLLMImageTag             = "trtllm-node"
	defaultTRTLLMSourceImage          = "nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3"
	defaultNVIDIAImageTag             = "nvcr.io/nvidia/vllm:26.01-py3"
	defaultNVIDIASourceImage          = "nvcr.io/nvidia/vllm:26.01-py3"
	trtllmDeploymentGuideURL          = "https://build.nvidia.com/spark/trt-llm/stacked-sparks"
	nvidiaDeploymentGuideURL          = "https://nvidia.github.io/spark-rapids-docs/"
	recipeMetadataKey                 = "recipe_ui"
	recipeMetadataManagedField        = "managed"
	nvcrProxyAuthURL                  = "https://nvcr.io/proxy_auth?scope=repository:nvidia/tensorrt-llm/release:pull"
	nvcrTagsListURL                   = "https://nvcr.io/v2/nvidia/tensorrt-llm/release/tags/list?n=2000"
	nvidiaNGCAPIURL                   = "https://catalog.ngc.nvidia.com/api/v3/orgs/nvidia/containers/vllm/versions"
	nvidiaNGCBaseURL                  = "https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm"
)

var (
	recipeRunnerRe           = regexp.MustCompile(`(?:^|\s)(?:exec\s+)?(?:\$\{recipe_runner\}|[^\s'"]*run-recipe\.sh)\s+([^\s'"]+)`)
	recipeTpRe               = regexp.MustCompile(`(?:^|\s)--tp\s+([0-9]+)`)
	recipeNodesRe            = regexp.MustCompile(`(?:^|\s)-n\s+("?[^"\s]+"?|\$\{[^}]+\}|[^\s]+)`)
	trtllmSourceImageRe      = regexp.MustCompile(`(?m)^SOURCE_IMAGE="([^"]+)"`)
	trtllmTagVersionRe       = regexp.MustCompile(`^(\d+)\.(\d+)\.(\d+)(?:rc(\d+))?(?:\.post(\d+))?$`)
	recipesBackendOverrideMu sync.RWMutex
	recipesBackendOverride   string
)

type recipeCatalogMeta struct {
	Name        string         `yaml:"name"`
	Description string         `yaml:"description"`
	Model       string         `yaml:"model"`
	SoloOnly    bool           `yaml:"solo_only"`
	ClusterOnly bool           `yaml:"cluster_only"`
	Defaults    map[string]any `yaml:"defaults"`
}

type RecipeCatalogItem struct {
	ID                    string `json:"id"`
	Ref                   string `json:"ref"`
	Path                  string `json:"path"`
	Name                  string `json:"name"`
	Description           string `json:"description"`
	Model                 string `json:"model"`
	SoloOnly              bool   `json:"soloOnly"`
	ClusterOnly           bool   `json:"clusterOnly"`
	DefaultTensorParallel int    `json:"defaultTensorParallel"`
}

type RecipeManagedModel struct {
	ModelID               string   `json:"modelId"`
	RecipeRef             string   `json:"recipeRef"`
	Name                  string   `json:"name"`
	Description           string   `json:"description"`
	Aliases               []string `json:"aliases"`
	UseModelName          string   `json:"useModelName"`
	Mode                  string   `json:"mode"` // solo|cluster
	TensorParallel        int      `json:"tensorParallel"`
	Nodes                 string   `json:"nodes,omitempty"`
	ExtraArgs             string   `json:"extraArgs,omitempty"`
	Group                 string   `json:"group"`
	Unlisted              bool     `json:"unlisted,omitempty"`
	Managed               bool     `json:"managed"`
	BenchyTrustRemoteCode *bool    `json:"benchyTrustRemoteCode,omitempty"`
}

type RecipeUIState struct {
	ConfigPath string               `json:"configPath"`
	BackendDir string               `json:"backendDir"`
	Recipes    []RecipeCatalogItem  `json:"recipes"`
	Models     []RecipeManagedModel `json:"models"`
	Groups     []string             `json:"groups"`
}

type RecipeBackendState struct {
	BackendDir         string                    `json:"backendDir"`
	BackendSource      string                    `json:"backendSource"`
	Options            []string                  `json:"options"`
	BackendKind        string                    `json:"backendKind"`
	BackendVendor      string                    `json:"backendVendor,omitempty"`
	DeploymentGuideURL string                    `json:"deploymentGuideUrl,omitempty"`
	RepoURL            string                    `json:"repoUrl,omitempty"`
	Actions            []RecipeBackendActionInfo `json:"actions"`
	TRTLLMImage        *RecipeBackendTRTLLMImage `json:"trtllmImage,omitempty"`
	NVIDIAImage        *RecipeBackendNVIDIAImage `json:"nvidiaImage,omitempty"`
}

type RecipeBackendActionInfo struct {
	Action      string `json:"action"`
	Label       string `json:"label"`
	CommandHint string `json:"commandHint,omitempty"`
}

type RecipeBackendTRTLLMImage struct {
	Selected        string   `json:"selected"`
	Default         string   `json:"default"`
	Latest          string   `json:"latest,omitempty"`
	UpdateAvailable bool     `json:"updateAvailable,omitempty"`
	Available       []string `json:"available,omitempty"`
	Warning         string   `json:"warning,omitempty"`
}

type RecipeBackendNVIDIAImage struct {
	Selected        string   `json:"selected"`
	Default         string   `json:"default"`
	Latest          string   `json:"latest,omitempty"`
	UpdateAvailable bool     `json:"updateAvailable,omitempty"`
	Available       []string `json:"available,omitempty"`
	Warning         string   `json:"warning,omitempty"`
}

type upsertRecipeModelRequest struct {
	ModelID               string   `json:"modelId"`
	RecipeRef             string   `json:"recipeRef"`
	Name                  string   `json:"name"`
	Description           string   `json:"description"`
	Aliases               []string `json:"aliases"`
	UseModelName          string   `json:"useModelName"`
	Mode                  string   `json:"mode"` // solo|cluster
	TensorParallel        int      `json:"tensorParallel"`
	Nodes                 string   `json:"nodes,omitempty"`
	ExtraArgs             string   `json:"extraArgs,omitempty"`
	Group                 string   `json:"group"`
	Unlisted              bool     `json:"unlisted,omitempty"`
	BenchyTrustRemoteCode *bool    `json:"benchyTrustRemoteCode,omitempty"`
}

type setRecipeBackendRequest struct {
	BackendDir string `json:"backendDir"`
}

type recipeBackendActionRequest struct {
	Action      string `json:"action"`
	SourceImage string `json:"sourceImage,omitempty"`
}

type recipeBackendActionResponse struct {
	Action     string `json:"action"`
	BackendDir string `json:"backendDir"`
	Command    string `json:"command"`
	Message    string `json:"message"`
	Output     string `json:"output,omitempty"`
	DurationMs int64  `json:"durationMs"`
}

func (pm *ProxyManager) apiGetRecipeState(c *gin.Context) {
	state, err := pm.buildRecipeUIState()
	if err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, state)
}

func (pm *ProxyManager) apiGetRecipeBackend(c *gin.Context) {
	c.JSON(http.StatusOK, pm.recipeBackendState())
}

func (pm *ProxyManager) apiSetRecipeBackend(c *gin.Context) {
	var req setRecipeBackendRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON body"})
		return
	}

	previousKind := detectRecipeBackendKind(recipesBackendDir())
	desired := expandLeadingTilde(strings.TrimSpace(req.BackendDir))
	if desired != "" {
		abs, err := filepath.Abs(desired)
		if err == nil {
			desired = abs
		}
		if stat, err := os.Stat(desired); err != nil || !stat.IsDir() {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("backend dir not found: %s", desired)})
			return
		}
		if _, err := os.Stat(filepath.Join(desired, "run-recipe.sh")); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("run-recipe.sh not found in backend: %s", desired)})
			return
		}
		if _, err := os.Stat(filepath.Join(desired, "recipes")); err != nil {
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("recipes dir not found in backend: %s", desired)})
			return
		}
	}

	setRecipesBackendOverride(desired)
	if err := pm.persistRecipesBackendOverride(desired); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	nextKind := detectRecipeBackendKind(recipesBackendDir())
	if err := pm.switchRecipeBackendConfig(previousKind, nextKind); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if err := pm.syncRecipeBackendMacros(); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}
	if err := pm.persistActiveConfigForBackend(nextKind); err != nil {
		c.JSON(http.StatusInternalServerError, gin.H{"error": err.Error()})
		return
	}

	if nextKind == "trtllm" {
		backendDir := strings.TrimSpace(recipesBackendDir())
		if backendDir != "" {
			image := resolveTRTLLMSourceImage(backendDir, "")
			if err := persistTRTLLMSourceImage(backendDir, image); err != nil {
				pm.proxyLogger.Warnf("failed to persist trtllm source image override dir=%s err=%v", backendDir, err)
			}
		}
	}

	if nextKind == "nvidia" {
		backendDir := strings.TrimSpace(recipesBackendDir())
		if backendDir != "" {
			image := resolveNVIDIASourceImage(backendDir, "")
			if err := persistNVIDIASourceImage(backendDir, image); err != nil {
				pm.proxyLogger.Warnf("failed to persist nvidia source image override dir=%s err=%v", backendDir, err)
			}
		}
	}

	c.JSON(http.StatusOK, pm.recipeBackendState())
}

func (pm *ProxyManager) apiRunRecipeBackendAction(c *gin.Context) {
	var req recipeBackendActionRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON body"})
		return
	}

	action := strings.ToLower(strings.TrimSpace(req.Action))
	backendDir := strings.TrimSpace(recipesBackendDir())
	if backendDir == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "backend dir is empty"})
		return
	}
	if stat, err := os.Stat(backendDir); err != nil || !stat.IsDir() {
		c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("backend dir not found: %s", backendDir)})
		return
	}

	backendKind := detectRecipeBackendKind(backendDir)
	hasGit := backendHasGitRepo(backendDir)
	hasBuildScript := backendHasBuildScript(backendDir)

	var cmd *exec.Cmd
	var commandText string
	var trtllmSourceImage string

	switch action {
	case "git_pull":
		if !hasGit {
			c.JSON(http.StatusBadRequest, gin.H{"error": "git actions are not available for this backend"})
			return
		}
		commandText = "git pull --ff-only"
		cmd = exec.CommandContext(c.Request.Context(), "git", "-C", backendDir, "pull", "--ff-only")
	case "git_pull_rebase":
		if !hasGit {
			c.JSON(http.StatusBadRequest, gin.H{"error": "git actions are not available for this backend"})
			return
		}
		commandText = "git pull --rebase --autostash"
		cmd = exec.CommandContext(c.Request.Context(), "git", "-C", backendDir, "pull", "--rebase", "--autostash")
	case "build_vllm":
		if backendKind == "trtllm" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "build_vllm is not supported for TRT-LLM backend"})
			return
		}
		if !hasBuildScript {
			script := filepath.Join(backendDir, "build-and-copy.sh")
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("build script not found: %s", script)})
			return
		}
		commandText = "./build-and-copy.sh --rebuild-deps --rebuild-vllm -c"
		ctx, cancel := context.WithTimeout(c.Request.Context(), 8*time.Hour)
		defer cancel()
		cmd = exec.CommandContext(ctx, "bash", "./build-and-copy.sh", "--rebuild-deps", "--rebuild-vllm", "-c")
		cmd.Dir = backendDir
	case "build_mxfp4":
		if backendKind == "trtllm" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "build_mxfp4 is not supported for TRT-LLM backend"})
			return
		}
		if !hasBuildScript {
			script := filepath.Join(backendDir, "build-and-copy.sh")
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("build script not found: %s", script)})
			return
		}
		commandText = "./build-and-copy.sh -t vllm-node-mxfp4 --rebuild-deps --rebuild-vllm --exp-mxfp4 -c"
		ctx, cancel := context.WithTimeout(c.Request.Context(), 8*time.Hour)
		defer cancel()
		cmd = exec.CommandContext(
			ctx,
			"bash",
			"./build-and-copy.sh",
			"-t", "vllm-node-mxfp4",
			"--rebuild-deps",
			"--rebuild-vllm",
			"--exp-mxfp4",
			"-c",
		)
		cmd.Dir = backendDir
	case "build_trtllm_image":
		if backendKind != "trtllm" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "build_trtllm_image is only supported for TRT-LLM backend"})
			return
		}
		if !hasBuildScript {
			script := filepath.Join(backendDir, "build-and-copy.sh")
			c.JSON(http.StatusBadRequest, gin.H{"error": fmt.Sprintf("build script not found: %s", script)})
			return
		}
		trtllmSourceImage = resolveTRTLLMSourceImage(backendDir, req.SourceImage)
		if trtllmSourceImage == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "source image is empty"})
			return
		}
		commandText = fmt.Sprintf("./build-and-copy.sh -t %s --source-image %s -c", defaultTRTLLMImageTag, trtllmSourceImage)
		ctx, cancel := context.WithTimeout(c.Request.Context(), 8*time.Hour)
		defer cancel()
		cmd = exec.CommandContext(
			ctx,
			"bash",
			"./build-and-copy.sh",
			"-t", defaultTRTLLMImageTag,
			"--source-image", trtllmSourceImage,
			"-c",
		)
		cmd.Dir = backendDir
	case "pull_trtllm_image":
		if backendKind != "trtllm" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "pull_trtllm_image is only supported for TRT-LLM backend"})
			return
		}
		trtllmImage := resolveTRTLLMSourceImage(backendDir, req.SourceImage)
		if trtllmImage == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "source image is empty"})
			return
		}
		commandText = fmt.Sprintf("docker pull %s", trtllmImage)
		ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Minute)
		defer cancel()
		cmd = exec.CommandContext(ctx, "docker", "pull", trtllmImage)
	case "update_trtllm_image":
		if backendKind != "trtllm" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "update_trtllm_image is only supported for TRT-LLM backend"})
			return
		}
		trtllmImage := resolveTRTLLMSourceImage(backendDir, req.SourceImage)
		if trtllmImage == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "source image is empty"})
			return
		}
		// If no source image specified, automatically use the latest version
		if req.SourceImage == "" {
			state := pm.recipeBackendState()
			if state.BackendKind == "trtllm" && state.TRTLLMImage != nil && state.TRTLLMImage.Latest != "" {
				trtllmImage = state.TRTLLMImage.Latest
			}
		}
		if err := persistTRTLLMSourceImage(backendDir, trtllmImage); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to persist trtllm image: %v", err)})
			return
		}
		commandText = fmt.Sprintf("docker pull %s && ./copy-image-to-peers.sh %s", trtllmImage, trtllmImage)
		ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Minute)
		defer cancel()
		cmd = exec.CommandContext(ctx, "bash", "-lc", fmt.Sprintf("docker pull %s && if [ -f ./copy-image-to-peers.sh ]; then ./copy-image-to-peers.sh %s; fi", trtllmImage, trtllmImage))
	case "pull_nvidia_image":
		if backendKind != "nvidia" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "pull_nvidia_image is only supported for NVIDIA backend"})
			return
		}
		nvidiaImage := resolveNVIDIASourceImage(backendDir, req.SourceImage)
		if nvidiaImage == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "source image is empty"})
			return
		}
		commandText = fmt.Sprintf("docker pull %s", nvidiaImage)
		ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Minute)
		defer cancel()
		cmd = exec.CommandContext(ctx, "docker", "pull", nvidiaImage)
	case "update_nvidia_image":
		if backendKind != "nvidia" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "update_nvidia_image is only supported for NVIDIA backend"})
			return
		}
		nvidiaImage := resolveNVIDIASourceImage(backendDir, req.SourceImage)
		if nvidiaImage == "" {
			c.JSON(http.StatusBadRequest, gin.H{"error": "source image is empty"})
			return
		}
		// If no source image specified, automatically use the latest version
		if req.SourceImage == "" {
			ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
			defer cancel()
			tags, err := fetchNVIDIAReleaseTags(ctx)
			if err == nil && len(tags) > 0 {
				latestImage := latestNVIDIATag(tags)
				if latestImage != "" {
					nvidiaImage = latestImage
				}
			}
		}
		if err := persistNVIDIASourceImage(backendDir, nvidiaImage); err != nil {
			c.JSON(http.StatusInternalServerError, gin.H{"error": fmt.Sprintf("failed to persist nvidia image: %v", err)})
			return
		}
		commandText = fmt.Sprintf("docker pull %s", nvidiaImage)
		ctx, cancel := context.WithTimeout(c.Request.Context(), 30*time.Minute)
		defer cancel()
		cmd = exec.CommandContext(ctx, "docker", "pull", nvidiaImage)
	default:
		c.JSON(http.StatusBadRequest, gin.H{"error": "unsupported action"})
		return
	}

	started := time.Now()
	output, err := cmd.CombinedOutput()
	durationMs := time.Since(started).Milliseconds()
	outputText := strings.TrimSpace(string(output))
	outputText = tailString(outputText, 120000)

	if err != nil {
		pm.proxyLogger.Errorf("backend action failed action=%s dir=%s err=%v", action, backendDir, err)
		errMsg := fmt.Sprintf("action %s failed: %v", action, err)
		if action == "git_pull" {
			lowerOut := strings.ToLower(outputText)
			if strings.Contains(lowerOut, "diverging branches") || strings.Contains(lowerOut, "can't be fast-forwarded") {
				errMsg = "action git_pull failed: backend has diverging history. Use 'Git Pull Rebase' to reconcile local commits with upstream."
			}
		}
		c.JSON(http.StatusBadGateway, gin.H{
			"error":      errMsg,
			"action":     action,
			"backendDir": backendDir,
			"command":    commandText,
			"output":     outputText,
			"durationMs": durationMs,
		})
		return
	}

	if action == "build_trtllm_image" || action == "update_trtllm_image" {
		_ = persistTRTLLMSourceImage(backendDir, trtllmSourceImage)
	}

	pm.proxyLogger.Infof("backend action completed action=%s dir=%s durationMs=%d", action, backendDir, durationMs)
	c.JSON(http.StatusOK, recipeBackendActionResponse{
		Action:     action,
		BackendDir: backendDir,
		Command:    commandText,
		Message:    fmt.Sprintf("Action %s completed successfully.", action),
		Output:     outputText,
		DurationMs: durationMs,
	})
}

func (pm *ProxyManager) apiUpsertRecipeModel(c *gin.Context) {
	var req upsertRecipeModelRequest
	if err := c.ShouldBindJSON(&req); err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": "invalid JSON body"})
		return
	}

	state, err := pm.upsertRecipeModel(req)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, state)
}

func (pm *ProxyManager) apiDeleteRecipeModel(c *gin.Context) {
	modelID := strings.TrimSpace(c.Param("id"))
	if modelID == "" {
		c.JSON(http.StatusBadRequest, gin.H{"error": "model id is required"})
		return
	}

	state, err := pm.deleteRecipeModel(modelID)
	if err != nil {
		c.JSON(http.StatusBadRequest, gin.H{"error": err.Error()})
		return
	}
	c.JSON(http.StatusOK, state)
}

func (pm *ProxyManager) buildRecipeUIState() (RecipeUIState, error) {
	configPath, err := pm.getConfigPath()
	if err != nil {
		return RecipeUIState{}, err
	}

	backendDir := recipesBackendDir()
	catalog, catalogByID, err := loadRecipeCatalog(backendDir)
	if err != nil {
		return RecipeUIState{}, err
	}

	root, err := loadConfigRawMap(configPath)
	if err != nil {
		return RecipeUIState{}, err
	}

	modelsMap := getMap(root, "models")
	groupsMap := getMap(root, "groups")

	models := make([]RecipeManagedModel, 0, len(modelsMap))
	for modelID, raw := range modelsMap {
		modelMap, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if !recipeEntryTargetsActiveBackend(getMap(modelMap, "metadata"), backendDir) {
			continue
		}
		rm, ok := toRecipeManagedModel(modelID, modelMap, groupsMap)
		if ok && recipeManagedModelInCatalog(rm, catalogByID) {
			models = append(models, rm)
		}
	}
	sort.Slice(models, func(i, j int) bool { return models[i].ModelID < models[j].ModelID })

	groupNames := sortedGroupNames(groupsMap)
	return RecipeUIState{
		ConfigPath: configPath,
		BackendDir: backendDir,
		Recipes:    catalog,
		Models:     models,
		Groups:     groupNames,
	}, nil
}

func (pm *ProxyManager) recipeBackendState() RecipeBackendState {
	current, source := recipesBackendDirWithSource()
	options := recommendedRecipesBackendOptions()
	options = appendUniquePath(options, current)
	sort.Strings(options)

	kind := detectRecipeBackendKind(current)
	repoURL := backendGitRemoteOrigin(current)
	actions := recipeBackendActionsForKind(kind, current, repoURL)

	state := RecipeBackendState{
		BackendDir:    current,
		BackendSource: source,
		Options:       options,
		BackendKind:   kind,
		BackendVendor: recipeBackendVendor(kind),
		RepoURL:       repoURL,
		Actions:       actions,
	}
	if kind == "trtllm" {
		state.TRTLLMImage = buildTRTLLMImageState(current)
		state.DeploymentGuideURL = trtllmDeploymentGuideURL
	}
	if kind == "nvidia" {
		state.NVIDIAImage = buildNVIDIAImageState(current)
		state.DeploymentGuideURL = nvidiaDeploymentGuideURL
	}
	return state
}

func (pm *ProxyManager) upsertRecipeModel(req upsertRecipeModelRequest) (RecipeUIState, error) {
	modelID := strings.TrimSpace(req.ModelID)
	if modelID == "" {
		return RecipeUIState{}, errors.New("modelId is required")
	}
	recipeRefInput := strings.TrimSpace(req.RecipeRef)
	if recipeRefInput == "" {
		return RecipeUIState{}, errors.New("recipeRef is required")
	}

	configPath, err := pm.getConfigPath()
	if err != nil {
		return RecipeUIState{}, err
	}

	catalog, catalogByID, err := loadRecipeCatalog(recipesBackendDir())
	if err != nil {
		return RecipeUIState{}, err
	}
	_ = catalog

	resolvedRecipeRef, catalogRecipe, err := resolveRecipeRef(recipeRefInput, catalogByID)
	if err != nil {
		return RecipeUIState{}, err
	}

	mode := strings.ToLower(strings.TrimSpace(req.Mode))
	if mode == "" {
		if catalogRecipe.SoloOnly {
			mode = "solo"
		} else {
			mode = "cluster"
		}
	}
	if mode != "solo" && mode != "cluster" {
		return RecipeUIState{}, errors.New("mode must be 'solo' or 'cluster'")
	}
	if catalogRecipe.SoloOnly && mode != "solo" {
		return RecipeUIState{}, fmt.Errorf("recipe %s requires solo mode", recipeRefInput)
	}
	if catalogRecipe.ClusterOnly && mode != "cluster" {
		return RecipeUIState{}, fmt.Errorf("recipe %s requires cluster mode", recipeRefInput)
	}

	tp := req.TensorParallel
	if tp <= 0 {
		tp = catalogRecipe.DefaultTensorParallel
	}
	if tp <= 0 {
		tp = 1
	}

	root, err := loadConfigRawMap(configPath)
	if err != nil {
		return RecipeUIState{}, err
	}
	ensureRecipeMacros(root, configPath)
	modelsMap := getMap(root, "models")
	groupsMap := getMap(root, "groups")

	nodes := strings.TrimSpace(req.Nodes)
	if mode == "cluster" && nodes == "" {
		if expr, ok := backendMacroExpr(root, "nodes"); ok {
			nodes = expr
		} else {
			return RecipeUIState{}, errors.New("nodes is required for cluster mode (backend nodes macro not found)")
		}
	}

	groupName := strings.TrimSpace(req.Group)
	if groupName == "" {
		groupName = defaultRecipeGroupName
	}

	name := strings.TrimSpace(req.Name)
	if name == "" {
		name = modelID
	}
	description := strings.TrimSpace(req.Description)
	if description == "" {
		description = catalogRecipe.Description
	}

	useModelName := strings.TrimSpace(req.UseModelName)
	if useModelName == "" {
		useModelName = catalogRecipe.Model
	}

	existing := getMap(modelsMap, modelID)
	modelEntry := cloneMap(existing)
	modelEntry["name"] = name
	modelEntry["description"] = description
	modelEntry["proxy"] = "http://127.0.0.1:${PORT}"
	modelEntry["checkEndpoint"] = "/health"
	modelEntry["ttl"] = 0
	modelEntry["useModelName"] = useModelName
	modelEntry["unlisted"] = req.Unlisted
	modelEntry["aliases"] = cleanAliases(req.Aliases)

	cmdStopExpr := "true"
	stopPrefix := ""
	if expr, ok := backendMacroExpr(root, "stop_cluster"); ok {
		cmdStopExpr = expr
		stopPrefix = expr + "; "
	}

	runner := filepath.Join(recipesBackendDir(), "run-recipe.sh")
	if hasMacro(root, "recipe_runner") {
		runner = "${recipe_runner}"
	}

	var cmdParts []string
	cmdParts = append(cmdParts, "bash -lc '", stopPrefix, "exec ", runner, " ", quoteForCommand(resolvedRecipeRef))
	if mode == "solo" {
		cmdParts = append(cmdParts, " --solo")
	} else {
		cmdParts = append(cmdParts, " -n ", quoteForCommand(nodes))
	}
	if tp > 0 {
		cmdParts = append(cmdParts, " --tp ", strconv.Itoa(tp))
	}
	cmdParts = append(cmdParts, " --port ${PORT}")
	if extra := strings.TrimSpace(req.ExtraArgs); extra != "" {
		cmdParts = append(cmdParts, " ", extra)
	}
	cmdParts = append(cmdParts, "'")

	modelEntry["cmd"] = strings.Join(cmdParts, "")
	modelEntry["cmdStop"] = fmt.Sprintf("bash -lc '%s'", cmdStopExpr)

	meta := getMap(existing, "metadata")
	if len(meta) == 0 {
		meta = map[string]any{}
	}
	meta[recipeMetadataKey] = map[string]any{
		recipeMetadataManagedField: true,
		"recipe_ref":               resolvedRecipeRef,
		"mode":                     mode,
		"tensor_parallel":          tp,
		"nodes":                    nodes,
		"extra_args":               strings.TrimSpace(req.ExtraArgs),
		"group":                    groupName,
		"backend_dir":              recipesBackendDir(),
	}
	if req.BenchyTrustRemoteCode != nil {
		benchyMeta := getMap(meta, "benchy")
		benchyMeta["trust_remote_code"] = *req.BenchyTrustRemoteCode
		meta["benchy"] = benchyMeta
	}
	modelEntry["metadata"] = meta

	modelsMap[modelID] = modelEntry
	root["models"] = modelsMap

	removeModelFromAllGroups(groupsMap, modelID)
	group := getMap(groupsMap, groupName)
	if _, ok := group["swap"]; !ok {
		group["swap"] = true
	}
	if _, ok := group["exclusive"]; !ok {
		group["exclusive"] = true
	}
	members := append(groupMembers(group), modelID)
	group["members"] = uniqueStrings(members)
	groupsMap[groupName] = group
	root["groups"] = groupsMap

	if err := writeConfigRawMap(configPath, root); err != nil {
		return RecipeUIState{}, err
	}

	if conf, err := config.LoadConfig(configPath); err == nil {
		pm.Lock()
		pm.config = conf
		pm.Unlock()
	}
	return pm.buildRecipeUIState()
}

func (pm *ProxyManager) deleteRecipeModel(modelID string) (RecipeUIState, error) {
	configPath, err := pm.getConfigPath()
	if err != nil {
		return RecipeUIState{}, err
	}

	root, err := loadConfigRawMap(configPath)
	if err != nil {
		return RecipeUIState{}, err
	}
	ensureRecipeMacros(root, configPath)

	modelsMap := getMap(root, "models")
	if _, ok := modelsMap[modelID]; !ok {
		return RecipeUIState{}, fmt.Errorf("model %s not found", modelID)
	}
	delete(modelsMap, modelID)
	root["models"] = modelsMap

	groupsMap := getMap(root, "groups")
	removeModelFromAllGroups(groupsMap, modelID)
	root["groups"] = groupsMap

	if err := writeConfigRawMap(configPath, root); err != nil {
		return RecipeUIState{}, err
	}

	if conf, err := config.LoadConfig(configPath); err == nil {
		pm.Lock()
		pm.config = conf
		pm.Unlock()
	}
	return pm.buildRecipeUIState()
}

func recipesBackendDir() string {
	dir, _ := recipesBackendDirWithSource()
	return dir
}

func recipesBackendDirWithSource() (string, string) {
	if v := strings.TrimSpace(getRecipesBackendOverride()); v != "" {
		return v, "override"
	}
	if v := strings.TrimSpace(os.Getenv(recipesBackendDirEnv)); v != "" {
		return v, "env"
	}
	if home := userHomeDir(); home != "" {
		return filepath.Join(home, defaultRecipesBackendSubdir), "default"
	}
	return defaultRecipesBackendSubdir, "default"
}

func localRecipesDir() string {
	if v := strings.TrimSpace(os.Getenv(recipesLocalDirEnv)); v != "" {
		return v
	}
	if home := userHomeDir(); home != "" {
		return filepath.Join(home, "llama-swap", "recipes")
	}
	return filepath.FromSlash(defaultRecipesLocalSubdir)
}

func userHomeDir() string {
	if v := strings.TrimSpace(os.Getenv("HOME")); v != "" {
		return v
	}
	if home, err := os.UserHomeDir(); err == nil {
		return strings.TrimSpace(home)
	}
	return ""
}

func setRecipesBackendOverride(path string) {
	recipesBackendOverrideMu.Lock()
	recipesBackendOverride = strings.TrimSpace(path)
	recipesBackendOverrideMu.Unlock()
}

func getRecipesBackendOverride() string {
	recipesBackendOverrideMu.RLock()
	defer recipesBackendOverrideMu.RUnlock()
	return recipesBackendOverride
}

func recommendedRecipesBackendOptions() []string {
	options := make([]string, 0, 6)
	if home := userHomeDir(); home != "" {
		options = append(options,
			filepath.Join(home, defaultRecipesBackendSubdir),
			filepath.Join(home, defaultRecipesBackendAltSubdir),
			filepath.Join(home, defaultRecipesBackendSQLSubdir),
			filepath.Join(home, defaultRecipesBackendNvidiaSubdir),
		)
	}
	if v := strings.TrimSpace(os.Getenv(recipesBackendDirEnv)); v != "" {
		options = append(options, v)
	}
	return uniqueExistingDirs(options)
}

func detectRecipeBackendKind(backendDir string) string {
	base := strings.ToLower(filepath.Base(strings.TrimSpace(backendDir)))
	switch {
	case strings.Contains(base, "trtllm"):
		return "trtllm"
	case strings.Contains(base, "sqlang"):
		return "sqlang"
	case strings.Contains(base, "vllm") && strings.Contains(base, "nvidia"):
		return "nvidia"
	case strings.Contains(base, "vllm"):
		return "vllm"
	default:
		return "custom"
	}
}

func backendHasGitRepo(backendDir string) bool {
	if backendDir == "" {
		return false
	}
	if _, err := os.Stat(filepath.Join(backendDir, ".git")); err == nil {
		return true
	}
	return false
}

func backendHasBuildScript(backendDir string) bool {
	if backendDir == "" {
		return false
	}
	if stat, err := os.Stat(filepath.Join(backendDir, "build-and-copy.sh")); err == nil {
		return !stat.IsDir()
	}
	return false
}

func backendGitRemoteOrigin(backendDir string) string {
	if !backendHasGitRepo(backendDir) {
		return ""
	}
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	out, err := exec.CommandContext(ctx, "git", "-C", backendDir, "config", "--get", "remote.origin.url").Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

func shortRepoLabel(repoURL string) string {
	repoURL = strings.TrimSpace(repoURL)
	repoURL = strings.TrimSuffix(repoURL, ".git")
	repoURL = strings.TrimPrefix(repoURL, "git@github.com:")
	repoURL = strings.TrimPrefix(repoURL, "https://github.com/")
	parts := strings.Split(repoURL, "/")
	if len(parts) >= 2 && parts[0] != "" && parts[1] != "" {
		return parts[0] + "/" + parts[1]
	}
	if repoURL == "" {
		return "origin"
	}
	return repoURL
}

func recipeBackendVendor(kind string) string {
	k := strings.ToLower(strings.TrimSpace(kind))
	if k == "trtllm" || k == "vllm_nvidia" || k == "nvidia" {
		return "nvidia"
	}
	return ""
}

func recipeBackendActionsForKind(kind, backendDir, repoURL string) []RecipeBackendActionInfo {
	actions := make([]RecipeBackendActionInfo, 0, 4)
	if backendHasGitRepo(backendDir) {
		repoLabel := shortRepoLabel(repoURL)
		actions = append(actions,
			RecipeBackendActionInfo{Action: "git_pull", Label: fmt.Sprintf("Git Pull (%s)", repoLabel), CommandHint: "git pull --ff-only"},
			RecipeBackendActionInfo{Action: "git_pull_rebase", Label: fmt.Sprintf("Git Pull Rebase (%s)", repoLabel), CommandHint: "git pull --rebase --autostash"},
		)
	}

	if kind == "nvidia" {
		// NVIDIA actions don't require build-and-copy.sh
		actions = append(actions,
			RecipeBackendActionInfo{
				Action:      "pull_nvidia_image",
				Label:       "Pull NVIDIA Image",
				CommandHint: "docker pull <selected>",
			},
			RecipeBackendActionInfo{
				Action:      "update_nvidia_image",
				Label:       "Update NVIDIA Image",
				CommandHint: "docker pull <selected> + persist as new default",
			},
		)
		return actions
	}

	if !backendHasBuildScript(backendDir) {
		return actions
	}

	if kind == "trtllm" {
		actions = append(actions,
			RecipeBackendActionInfo{
				Action:      "pull_trtllm_image",
				Label:       "Pull TRT-LLM Image",
				CommandHint: "docker pull <selected>",
			},
			RecipeBackendActionInfo{
				Action:      "update_trtllm_image",
				Label:       "Update TRT-LLM Image",
				CommandHint: "docker pull <selected> + persist + copy to peers",
			},
		)
		return actions
	}

	actions = append(actions,
		RecipeBackendActionInfo{Action: "build_vllm", Label: "Build vLLM", CommandHint: "./build-and-copy.sh --rebuild-deps --rebuild-vllm -c"},
		RecipeBackendActionInfo{Action: "build_mxfp4", Label: "Build MXFP4", CommandHint: "./build-and-copy.sh -t vllm-node-mxfp4 --rebuild-deps --rebuild-vllm --exp-mxfp4 -c"},
	)
	return actions
}

func trtllmSourceImageOverridePath(backendDir string) string {
	if strings.TrimSpace(backendDir) == "" {
		return ""
	}
	return filepath.Join(backendDir, trtllmSourceImageOverrideFile)
}

func loadTRTLLMSourceImage(backendDir string) string {
	override := trtllmSourceImageOverridePath(backendDir)
	if override != "" {
		if raw, err := os.ReadFile(override); err == nil {
			if v := strings.TrimSpace(string(raw)); v != "" {
				return v
			}
		}
	}
	return ""
}

func persistTRTLLMSourceImage(backendDir, image string) error {
	override := trtllmSourceImageOverridePath(backendDir)
	if override == "" {
		return nil
	}
	image = strings.TrimSpace(image)
	if image == "" {
		if err := os.Remove(override); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		return nil
	}
	tmp := override + ".tmp"
	if err := os.WriteFile(tmp, []byte(image+"\n"), 0600); err != nil {
		return err
	}
	return os.Rename(tmp, override)
}

func readDefaultTRTLLMSourceImage(backendDir string) string {
	if envImage := strings.TrimSpace(os.Getenv("LLAMA_SWAP_TRTLLM_SOURCE_IMAGE")); envImage != "" {
		return envImage
	}
	scriptPath := filepath.Join(strings.TrimSpace(backendDir), "build-and-copy.sh")
	raw, err := os.ReadFile(scriptPath)
	if err != nil {
		return defaultTRTLLMSourceImage
	}
	m := trtllmSourceImageRe.FindStringSubmatch(string(raw))
	if len(m) > 1 {
		if v := strings.TrimSpace(m[1]); v != "" {
			return v
		}
	}
	return defaultTRTLLMSourceImage
}

func resolveTRTLLMSourceImage(backendDir, requested string) string {
	if v := strings.TrimSpace(requested); v != "" {
		return v
	}
	if v := loadTRTLLMSourceImage(backendDir); v != "" {
		return v
	}
	return readDefaultTRTLLMSourceImage(backendDir)
}

func trtllmUpdateScript(imageTag string) string {
	imageTag = strings.TrimSpace(imageTag)
	if imageTag == "" {
		imageTag = defaultTRTLLMImageTag
	}

	return fmt.Sprintf(`set -euo pipefail

./build-and-copy.sh -t %s --source-image "${NEW_IMAGE}" -c

if [[ -z "${OLD_IMAGE:-}" || "${OLD_IMAGE}" == "${NEW_IMAGE}" ]]; then
  echo "No old TRT-LLM source image to remove."
  exit 0
fi

echo "Removing old TRT-LLM source image locally: ${OLD_IMAGE}"
docker image rm -f "${OLD_IMAGE}" || true

declare -a PEER_NODES=()
if [[ -f "./autodiscover.sh" ]]; then
  source "./autodiscover.sh" || true
  detect_nodes >/dev/null 2>&1 || true
fi

if [[ ${#PEER_NODES[@]} -eq 0 ]]; then
  echo "No peer nodes detected for old image cleanup."
  exit 0
fi

for host in "${PEER_NODES[@]}"; do
  [[ -n "${host}" ]] || continue
  echo "Removing old TRT-LLM source image on ${host}: ${OLD_IMAGE}"
  ssh -o BatchMode=yes -o ConnectTimeout=10 "${USER}@${host}" "docker image rm -f \"${OLD_IMAGE}\" || true" || true
done
`, quoteForCommand(imageTag))
}

type nvcrProxyAuthResponse struct {
	Token string `json:"token"`
}

type nvcrTagsResponse struct {
	Name string   `json:"name"`
	Tags []string `json:"tags"`
}

func fetchTRTLLMReleaseTags(ctx context.Context) ([]string, error) {
	authReq, err := http.NewRequestWithContext(ctx, http.MethodGet, nvcrProxyAuthURL, nil)
	if err != nil {
		return nil, err
	}
	authResp, err := http.DefaultClient.Do(authReq)
	if err != nil {
		return nil, err
	}
	defer authResp.Body.Close()
	if authResp.StatusCode < 200 || authResp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(authResp.Body, 2048))
		return nil, fmt.Errorf("nvcr auth failed: %s %s", authResp.Status, strings.TrimSpace(string(body)))
	}
	var auth nvcrProxyAuthResponse
	if err := json.NewDecoder(authResp.Body).Decode(&auth); err != nil {
		return nil, err
	}
	if strings.TrimSpace(auth.Token) == "" {
		return nil, errors.New("nvcr auth returned empty token")
	}

	tagsReq, err := http.NewRequestWithContext(ctx, http.MethodGet, nvcrTagsListURL, nil)
	if err != nil {
		return nil, err
	}
	tagsReq.Header.Set("Authorization", "Bearer "+auth.Token)

	tagsResp, err := http.DefaultClient.Do(tagsReq)
	if err != nil {
		return nil, err
	}
	defer tagsResp.Body.Close()
	if tagsResp.StatusCode < 200 || tagsResp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(tagsResp.Body, 2048))
		return nil, fmt.Errorf("nvcr tags failed: %s %s", tagsResp.Status, strings.TrimSpace(string(body)))
	}

	var payload nvcrTagsResponse
	if err := json.NewDecoder(tagsResp.Body).Decode(&payload); err != nil {
		return nil, err
	}
	return payload.Tags, nil
}

type trtllmTagVersion struct {
	Major int
	Minor int
	Patch int
	RC    *int
	Post  int
	Raw   string
}

func parseTRTLLMTagVersion(tag string) (trtllmTagVersion, bool) {
	tag = strings.TrimSpace(tag)
	m := trtllmTagVersionRe.FindStringSubmatch(tag)
	if len(m) == 0 {
		return trtllmTagVersion{}, false
	}
	major, _ := strconv.Atoi(m[1])
	minor, _ := strconv.Atoi(m[2])
	patch, _ := strconv.Atoi(m[3])
	var rcPtr *int
	if m[4] != "" {
		rc, _ := strconv.Atoi(m[4])
		rcPtr = &rc
	}
	post := 0
	if m[5] != "" {
		post, _ = strconv.Atoi(m[5])
	}
	return trtllmTagVersion{Major: major, Minor: minor, Patch: patch, RC: rcPtr, Post: post, Raw: tag}, true
}

func compareTRTLLMTagVersion(a, b trtllmTagVersion) int {
	if a.Major != b.Major {
		if a.Major < b.Major {
			return -1
		}
		return 1
	}
	if a.Minor != b.Minor {
		if a.Minor < b.Minor {
			return -1
		}
		return 1
	}
	if a.Patch != b.Patch {
		if a.Patch < b.Patch {
			return -1
		}
		return 1
	}

	if a.RC != nil && b.RC == nil {
		return -1
	}
	if a.RC == nil && b.RC != nil {
		return 1
	}
	if a.RC != nil && b.RC != nil {
		if *a.RC < *b.RC {
			return -1
		}
		if *a.RC > *b.RC {
			return 1
		}
	}
	if a.Post != b.Post {
		if a.Post < b.Post {
			return -1
		}
		return 1
	}
	return 0
}

func latestTRTLLMTag(tags []string) string {
	var best trtllmTagVersion
	hasBest := false
	for _, tag := range tags {
		v, ok := parseTRTLLMTagVersion(tag)
		if !ok {
			continue
		}
		if !hasBest || compareTRTLLMTagVersion(v, best) > 0 {
			best = v
			hasBest = true
		}
	}
	if !hasBest {
		return ""
	}
	return best.Raw
}

func topTRTLLMTags(tags []string, limit int) []string {
	versions := make([]trtllmTagVersion, 0, len(tags))
	for _, tag := range tags {
		v, ok := parseTRTLLMTagVersion(tag)
		if ok {
			versions = append(versions, v)
		}
	}
	sort.Slice(versions, func(i, j int) bool {
		return compareTRTLLMTagVersion(versions[i], versions[j]) > 0
	})
	if limit <= 0 || len(versions) < limit {
		limit = len(versions)
	}
	out := make([]string, 0, limit)
	for i := 0; i < limit; i++ {
		out = append(out, versions[i].Raw)
	}
	return out
}

func tagFromImageRef(image string) string {
	image = strings.TrimSpace(image)
	if image == "" {
		return ""
	}
	if idx := strings.LastIndex(image, ":"); idx >= 0 && idx < len(image)-1 {
		return image[idx+1:]
	}
	return ""
}

func appendUniqueString(items []string, value string) []string {
	value = strings.TrimSpace(value)
	if value == "" {
		return items
	}
	for _, item := range items {
		if item == value {
			return items
		}
	}
	return append(items, value)
}

func buildTRTLLMImageState(backendDir string) *RecipeBackendTRTLLMImage {
	defaultImage := readDefaultTRTLLMSourceImage(backendDir)
	selectedImage := resolveTRTLLMSourceImage(backendDir, "")
	if selectedImage == "" {
		selectedImage = defaultImage
	}
	state := &RecipeBackendTRTLLMImage{
		Selected: selectedImage,
		Default:  defaultImage,
	}
	state.Available = appendUniqueString(state.Available, selectedImage)
	state.Available = appendUniqueString(state.Available, defaultImage)

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	tags, err := fetchTRTLLMReleaseTags(ctx)
	if err != nil {
		state.Warning = fmt.Sprintf("No se pudieron consultar tags de nvcr.io: %v", err)
		return state
	}

	latestTag := latestTRTLLMTag(tags)
	if latestTag != "" {
		latestImage := "nvcr.io/nvidia/tensorrt-llm/release:" + latestTag
		state.Latest = latestImage
		state.Available = appendUniqueString(state.Available, latestImage)

		selectedTag := tagFromImageRef(selectedImage)
		selectedVersion, selectedOK := parseTRTLLMTagVersion(selectedTag)
		latestVersion, latestOK := parseTRTLLMTagVersion(latestTag)
		if selectedOK && latestOK && compareTRTLLMTagVersion(selectedVersion, latestVersion) < 0 {
			state.UpdateAvailable = true
		}
	}

	for _, tag := range topTRTLLMTags(tags, 12) {
		state.Available = appendUniqueString(state.Available, "nvcr.io/nvidia/tensorrt-llm/release:"+tag)
	}
	return state
}

func uniqueExistingDirs(paths []string) []string {
	seen := make(map[string]struct{}, len(paths))
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		p = strings.TrimSpace(p)
		if p == "" {
			continue
		}
		p = expandLeadingTilde(p)
		abs, err := filepath.Abs(p)
		if err == nil {
			p = abs
		}
		if _, ok := seen[p]; ok {
			continue
		}
		if stat, err := os.Stat(p); err == nil && stat.IsDir() {
			seen[p] = struct{}{}
			out = append(out, p)
		}
	}
	return out
}

func appendUniquePath(paths []string, candidate string) []string {
	candidate = strings.TrimSpace(candidate)
	if candidate == "" {
		return paths
	}
	absCandidate, err := filepath.Abs(candidate)
	if err == nil {
		candidate = absCandidate
	}
	for _, p := range paths {
		if p == candidate {
			return paths
		}
	}
	return append(paths, candidate)
}

func (pm *ProxyManager) recipesBackendOverrideFile() string {
	if v := strings.TrimSpace(os.Getenv(recipesBackendOverrideFileEnv)); v != "" {
		return expandLeadingTilde(v)
	}
	if pm != nil {
		if cfgPath := strings.TrimSpace(pm.configPath); cfgPath != "" {
			return filepath.Join(filepath.Dir(cfgPath), ".recipes_backend_dir")
		}
	}
	if home := userHomeDir(); home != "" {
		return filepath.Join(home, ".config", "llama-swap", "recipes_backend_dir")
	}
	return ""
}

func (pm *ProxyManager) loadRecipesBackendOverride() {
	path := pm.recipesBackendOverrideFile()
	if path == "" {
		return
	}
	raw, err := os.ReadFile(path)
	if err != nil {
		return
	}
	value := expandLeadingTilde(strings.TrimSpace(string(raw)))
	if value == "" {
		return
	}
	setRecipesBackendOverride(value)
}

func (pm *ProxyManager) persistRecipesBackendOverride(path string) error {
	filePath := pm.recipesBackendOverrideFile()
	if filePath == "" {
		return nil
	}

	path = strings.TrimSpace(path)
	if path == "" {
		if err := os.Remove(filePath); err != nil && !errors.Is(err, fs.ErrNotExist) {
			return err
		}
		return nil
	}

	parent := filepath.Dir(filePath)
	if parent != "" {
		if err := os.MkdirAll(parent, 0755); err != nil {
			return err
		}
	}

	tmp := filePath + ".tmp"
	if err := os.WriteFile(tmp, []byte(path+"\n"), 0600); err != nil {
		return err
	}
	return os.Rename(tmp, filePath)
}

func (pm *ProxyManager) getConfigPath() (string, error) {
	if v := strings.TrimSpace(pm.configPath); v != "" {
		return v, nil
	}
	if v := strings.TrimSpace(os.Getenv("LLAMA_SWAP_CONFIG_PATH")); v != "" {
		return v, nil
	}
	return "", errors.New("config path is unknown (start llama-swap with --config)")
}

func normalizeBackendConfigKind(kind string) string {
	switch strings.ToLower(strings.TrimSpace(kind)) {
	case "vllm", "vllm_nvidia", "trtllm", "sqlang":
		return strings.ToLower(strings.TrimSpace(kind))
	default:
		return "custom"
	}
}

func backendScopedConfigPath(configPath, backendKind string) string {
	configPath = strings.TrimSpace(configPath)
	if configPath == "" {
		return ""
	}
	ext := filepath.Ext(configPath)
	if ext == "" {
		ext = ".yaml"
	}
	base := strings.TrimSuffix(filepath.Base(configPath), ext)
	kind := normalizeBackendConfigKind(backendKind)
	return filepath.Join(filepath.Dir(configPath), fmt.Sprintf("%s.%s%s", base, kind, ext))
}

func copyFileAtomic(src, dst string) error {
	raw, err := os.ReadFile(src)
	if err != nil {
		return err
	}
	parent := filepath.Dir(dst)
	if parent != "" {
		if err := os.MkdirAll(parent, 0755); err != nil {
			return err
		}
	}
	tmp := dst + ".tmp"
	if err := os.WriteFile(tmp, raw, 0600); err != nil {
		return err
	}
	return os.Rename(tmp, dst)
}

func (pm *ProxyManager) switchRecipeBackendConfig(previousKind, nextKind string) error {
	configPath, err := pm.getConfigPath()
	if err != nil {
		return err
	}

	prevPath := backendScopedConfigPath(configPath, previousKind)
	nextPath := backendScopedConfigPath(configPath, nextKind)

	if prevPath != "" {
		if err := copyFileAtomic(configPath, prevPath); err != nil {
			return fmt.Errorf("failed to persist %s backend config: %w", normalizeBackendConfigKind(previousKind), err)
		}
	}

	if nextPath != "" && nextPath != prevPath {
		if _, err := os.Stat(nextPath); err == nil {
			if err := copyFileAtomic(nextPath, configPath); err != nil {
				return fmt.Errorf("failed to load %s backend config: %w", normalizeBackendConfigKind(nextKind), err)
			}
		} else if !errors.Is(err, fs.ErrNotExist) {
			return fmt.Errorf("failed to inspect %s backend config: %w", normalizeBackendConfigKind(nextKind), err)
		}
	}

	return nil
}

func (pm *ProxyManager) persistActiveConfigForBackend(kind string) error {
	configPath, err := pm.getConfigPath()
	if err != nil {
		return err
	}
	scopedPath := backendScopedConfigPath(configPath, kind)
	if scopedPath == "" {
		return nil
	}
	if err := copyFileAtomic(configPath, scopedPath); err != nil {
		return fmt.Errorf("failed to persist active %s backend config: %w", normalizeBackendConfigKind(kind), err)
	}
	return nil
}

func (pm *ProxyManager) syncRecipeBackendMacros() error {
	configPath, err := pm.getConfigPath()
	if err != nil {
		return err
	}

	root, err := loadConfigRawMap(configPath)
	if err != nil {
		return err
	}
	ensureRecipeMacros(root, configPath)
	if err := writeConfigRawMap(configPath, root); err != nil {
		return err
	}

	if conf, err := config.LoadConfig(configPath); err == nil {
		pm.Lock()
		pm.config = conf
		pm.Unlock()
	}
	return nil
}

func loadRecipeCatalog(backendDir string) ([]RecipeCatalogItem, map[string]RecipeCatalogItem, error) {
	recipesDir := filepath.Join(strings.TrimSpace(backendDir), "recipes")
	items := make([]RecipeCatalogItem, 0, 8)
	byID := make(map[string]RecipeCatalogItem)

	err := filepath.WalkDir(recipesDir, func(path string, d fs.DirEntry, walkErr error) error {
		if walkErr != nil {
			return walkErr
		}
		if d.IsDir() {
			return nil
		}
		ext := strings.ToLower(filepath.Ext(path))
		if ext != ".yaml" && ext != ".yml" {
			return nil
		}

		raw, err := os.ReadFile(path)
		if err != nil {
			return err
		}

		var meta recipeCatalogMeta
		if err := yaml.Unmarshal(raw, &meta); err != nil {
			return nil // skip malformed recipe files instead of failing whole API
		}

		base := filepath.Base(path)
		id := strings.TrimSuffix(strings.TrimSuffix(base, ".yaml"), ".yml")
		defaultTP := intFromAny(meta.Defaults["tensor_parallel"])
		if defaultTP <= 0 {
			defaultTP = 1
		}

		item := RecipeCatalogItem{
			ID:                    id,
			Ref:                   id,
			Path:                  path,
			Name:                  strings.TrimSpace(meta.Name),
			Description:           strings.TrimSpace(meta.Description),
			Model:                 strings.TrimSpace(meta.Model),
			SoloOnly:              meta.SoloOnly,
			ClusterOnly:           meta.ClusterOnly,
			DefaultTensorParallel: defaultTP,
		}
		items = append(items, item)
		byID[id] = item
		return nil
	})
	if err != nil && !errors.Is(err, fs.ErrNotExist) {
		return nil, nil, err
	}

	sort.Slice(items, func(i, j int) bool { return items[i].ID < items[j].ID })
	return items, byID, nil
}

func resolveRecipeRef(recipeRef string, catalogByID map[string]RecipeCatalogItem) (string, RecipeCatalogItem, error) {
	if item, ok := catalogByID[recipeRef]; ok {
		return item.Ref, item, nil
	}

	localRecipes := localRecipesDir()
	candidates := []string{
		recipeRef,
		filepath.Join(recipesBackendDir(), "recipes", recipeRef),
		filepath.Join(recipesBackendDir(), "recipes", recipeRef+".yaml"),
		filepath.Join(recipesBackendDir(), "recipes", recipeRef+".yml"),
		filepath.Join(localRecipes, recipeRef),
		filepath.Join(localRecipes, recipeRef+".yaml"),
		filepath.Join(localRecipes, recipeRef+".yml"),
	}
	for _, c := range candidates {
		if c == "" {
			continue
		}
		if stat, err := os.Stat(c); err == nil && !stat.IsDir() {
			item := RecipeCatalogItem{
				ID:   filepath.Base(strings.TrimSuffix(strings.TrimSuffix(c, ".yaml"), ".yml")),
				Ref:  c,
				Path: c,
				Name: filepath.Base(c),
			}
			return c, item, nil
		}
	}
	return "", RecipeCatalogItem{}, fmt.Errorf("recipeRef not found: %s", recipeRef)
}

func loadConfigRawMap(configPath string) (map[string]any, error) {
	raw, err := os.ReadFile(configPath)
	if err != nil {
		return nil, err
	}
	var parsed any
	if err := yaml.Unmarshal(raw, &parsed); err != nil {
		return nil, err
	}
	normalized := normalizeYAMLValue(parsed)
	root, ok := normalized.(map[string]any)
	if !ok || root == nil {
		return map[string]any{}, nil
	}
	return root, nil
}

func writeConfigRawMap(configPath string, root map[string]any) error {
	rendered, err := yaml.Marshal(root)
	if err != nil {
		return err
	}
	if _, err := config.LoadConfigFromReader(bytes.NewReader(rendered)); err != nil {
		return fmt.Errorf("generated config is invalid: %w", err)
	}

	tmp := configPath + ".tmp"
	if err := os.WriteFile(tmp, rendered, 0600); err != nil {
		return err
	}
	return os.Rename(tmp, configPath)
}

func normalizeYAMLValue(v any) any {
	switch t := v.(type) {
	case map[string]any:
		m := make(map[string]any, len(t))
		for k, vv := range t {
			m[k] = normalizeYAMLValue(vv)
		}
		return m
	case map[any]any:
		m := make(map[string]any, len(t))
		for k, vv := range t {
			m[fmt.Sprintf("%v", k)] = normalizeYAMLValue(vv)
		}
		return m
	case []any:
		out := make([]any, 0, len(t))
		for _, item := range t {
			out = append(out, normalizeYAMLValue(item))
		}
		return out
	default:
		return v
	}
}

func getMap(parent map[string]any, key string) map[string]any {
	if parent == nil {
		return map[string]any{}
	}
	if key == "" {
		return parent
	}
	if raw, ok := parent[key]; ok {
		if m, ok := raw.(map[string]any); ok {
			return m
		}
	}
	m := map[string]any{}
	parent[key] = m
	return m
}

func cloneMap(in map[string]any) map[string]any {
	if in == nil {
		return map[string]any{}
	}
	out := make(map[string]any, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func getString(m map[string]any, key string) string {
	if m == nil {
		return ""
	}
	if raw, ok := m[key]; ok {
		return strings.TrimSpace(fmt.Sprintf("%v", raw))
	}
	return ""
}

func getBool(m map[string]any, key string) bool {
	if m == nil {
		return false
	}
	v, ok := m[key]
	if !ok {
		return false
	}
	b, parsed := parseAnyBool(v)
	return parsed && b
}

func hasMacro(root map[string]any, name string) bool {
	macros := getMap(root, "macros")
	_, ok := macros[name]
	return ok
}

func backendMacroExpr(root map[string]any, suffix string) (string, bool) {
	suffix = strings.TrimSpace(suffix)
	if suffix == "" {
		return "", false
	}

	kind := detectRecipeBackendKind(recipesBackendDir())
	candidates := make([]string, 0, 8)
	seen := map[string]struct{}{}
	add := func(name string) {
		name = strings.TrimSpace(name)
		if name == "" {
			return
		}
		if _, ok := seen[name]; ok {
			return
		}
		seen[name] = struct{}{}
		candidates = append(candidates, name)
	}

	switch kind {
	case "trtllm":
		add("trtllm_" + suffix)
		add("vllm_" + suffix)
		add("sqlang_" + suffix)
	case "sqlang":
		add("sqlang_" + suffix)
		add("vllm_" + suffix)
		add("trtllm_" + suffix)
	case "vllm_nvidia":
		add("vllm_nvidia_" + suffix)
		add("vllm_" + suffix)
		add("trtllm_" + suffix)
		add("sqlang_" + suffix)
	default:
		add("vllm_" + suffix)
		add("trtllm_" + suffix)
		add("sqlang_" + suffix)
	}
	add(suffix)

	for _, name := range candidates {
		if hasMacro(root, name) {
			return "${" + name + "}", true
		}
	}
	return "", false
}

func ensureRecipeMacros(root map[string]any, configPath string) {
	macros := getMap(root, "macros")

	if _, ok := macros["user_home"]; !ok {
		macros["user_home"] = "${env.HOME}"
	}

	backendDir := strings.TrimSpace(recipesBackendDir())
	if backendDir != "" {
		backendDir = expandLeadingTilde(backendDir)
		if abs, err := filepath.Abs(backendDir); err == nil {
			backendDir = abs
		}

		// Keep these as concrete paths so config validation is stable regardless
		// of YAML map key ordering when the file is regenerated.
		macros["spark_root"] = backendDir
		macros["recipe_runner"] = filepath.Join(backendDir, "run-recipe.sh")
	}

	cfgPath := strings.TrimSpace(configPath)
	if cfgPath != "" {
		llamaRoot := filepath.Dir(cfgPath)
		if abs, err := filepath.Abs(llamaRoot); err == nil {
			llamaRoot = abs
		}
		macros["llama_root"] = llamaRoot
	} else if _, ok := macros["llama_root"]; !ok {
		macros["llama_root"] = "${user_home}/llama-swap"
	}

	kind := detectRecipeBackendKind(backendDir)
	if kind == "vllm_nvidia" {
		if _, ok := macros["vllm_nvidia_nodes"]; !ok {
			if v := strings.TrimSpace(fmt.Sprintf("%v", macros["vllm_nodes"])); v != "" && v != "<nil>" {
				macros["vllm_nvidia_nodes"] = v
			} else if head := strings.TrimSpace(fmt.Sprintf("%v", macros["vllm_head_ip"])); head != "" && head != "<nil>" {
				macros["vllm_nvidia_nodes"] = head
			}
		}
		if _, ok := macros["vllm_nvidia_stop_cluster"]; !ok {
			macros["vllm_nvidia_stop_cluster"] = "${recipe_runner} --stop -n ${vllm_nvidia_nodes}"
		}
	}

	root["macros"] = macros
}

func groupMembers(group map[string]any) []string {
	raw, ok := group["members"]
	if !ok {
		return nil
	}
	arr, ok := raw.([]any)
	if !ok {
		return nil
	}
	out := make([]string, 0, len(arr))
	for _, item := range arr {
		s := strings.TrimSpace(fmt.Sprintf("%v", item))
		if s != "" {
			out = append(out, s)
		}
	}
	return out
}

func removeModelFromAllGroups(groups map[string]any, modelID string) {
	for groupName, raw := range groups {
		group, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		members := groupMembers(group)
		filtered := make([]string, 0, len(members))
		for _, m := range members {
			if m != modelID {
				filtered = append(filtered, m)
			}
		}
		group["members"] = toAnySlice(filtered)
		groups[groupName] = group
	}
}

func sortedGroupNames(groups map[string]any) []string {
	names := make([]string, 0, len(groups))
	for groupName := range groups {
		names = append(names, groupName)
	}
	sort.Strings(names)
	return names
}

func toAnySlice(items []string) []any {
	out := make([]any, 0, len(items))
	for _, item := range items {
		out = append(out, item)
	}
	return out
}

func uniqueStrings(items []string) []any {
	seen := make(map[string]struct{}, len(items))
	out := make([]string, 0, len(items))
	for _, item := range items {
		s := strings.TrimSpace(item)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	sort.Strings(out)
	return toAnySlice(out)
}

func canonicalRecipeBackendDir(path string) string {
	path = strings.TrimSpace(path)
	if path == "" {
		return ""
	}
	path = expandLeadingTilde(path)
	if abs, err := filepath.Abs(path); err == nil {
		path = abs
	}
	return filepath.Clean(path)
}

func recipeEntryTargetsActiveBackend(metadata map[string]any, activeBackendDir string) bool {
	active := canonicalRecipeBackendDir(activeBackendDir)
	if active == "" {
		return true
	}
	recipeMeta := getMap(metadata, recipeMetadataKey)
	backendDir := canonicalRecipeBackendDir(getString(recipeMeta, "backend_dir"))
	if backendDir == "" {
		return true
	}
	return backendDir == active
}

func recipeManagedModelInCatalog(model RecipeManagedModel, catalogByID map[string]RecipeCatalogItem) bool {
	if len(catalogByID) == 0 {
		return true
	}
	ref := strings.TrimSpace(model.RecipeRef)
	if ref == "" {
		return true
	}
	_, ok := catalogByID[ref]
	return ok
}

func cleanAliases(aliases []string) []string {
	seen := make(map[string]struct{}, len(aliases))
	out := make([]string, 0, len(aliases))
	for _, alias := range aliases {
		s := strings.TrimSpace(alias)
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	sort.Strings(out)
	return out
}

func toRecipeManagedModel(modelID string, modelMap, groupsMap map[string]any) (RecipeManagedModel, bool) {
	cmd := getString(modelMap, "cmd")
	metadata := getMap(modelMap, "metadata")
	recipeMeta := getMap(metadata, recipeMetadataKey)
	managed := getBool(recipeMeta, recipeMetadataManagedField)

	isRecipeModel := recipeRunnerRe.MatchString(cmd)
	if !managed && !isRecipeModel {
		return RecipeManagedModel{}, false
	}

	recipeRef := getString(recipeMeta, "recipe_ref")
	if recipeRef == "" && cmd != "" {
		if m := recipeRunnerRe.FindStringSubmatch(cmd); len(m) > 1 {
			recipeRef = strings.TrimSpace(m[1])
		}
	}

	mode := getString(recipeMeta, "mode")
	if mode == "" {
		if strings.Contains(cmd, "--solo") {
			mode = "solo"
		} else {
			mode = "cluster"
		}
	}

	tp := intFromAny(recipeMeta["tensor_parallel"])
	if tp <= 0 && cmd != "" {
		if m := recipeTpRe.FindStringSubmatch(cmd); len(m) > 1 {
			tp, _ = strconv.Atoi(m[1])
		}
	}
	if tp <= 0 {
		tp = 1
	}

	nodes := getString(recipeMeta, "nodes")
	if nodes == "" && cmd != "" {
		if m := recipeNodesRe.FindStringSubmatch(cmd); len(m) > 1 {
			nodes = strings.Trim(m[1], `"`)
		}
	}

	groupName := getString(recipeMeta, "group")
	if groupName == "" {
		groupName = findModelGroup(modelID, groupsMap)
	}
	if groupName == "" {
		groupName = defaultRecipeGroupName
	}

	aliases := make([]string, 0)
	if rawAliases, ok := modelMap["aliases"].([]any); ok {
		for _, a := range rawAliases {
			s := strings.TrimSpace(fmt.Sprintf("%v", a))
			if s != "" {
				aliases = append(aliases, s)
			}
		}
	}

	var benchyTrustRemoteCode *bool
	if benchy := getMap(metadata, "benchy"); len(benchy) > 0 {
		if v, ok := benchy["trust_remote_code"]; ok {
			if parsed, ok := parseAnyBool(v); ok {
				benchyTrustRemoteCode = &parsed
			}
		}
	}

	return RecipeManagedModel{
		ModelID:               modelID,
		RecipeRef:             recipeRef,
		Name:                  getString(modelMap, "name"),
		Description:           getString(modelMap, "description"),
		Aliases:               aliases,
		UseModelName:          getString(modelMap, "useModelName"),
		Mode:                  mode,
		TensorParallel:        tp,
		Nodes:                 nodes,
		ExtraArgs:             getString(recipeMeta, "extra_args"),
		Group:                 groupName,
		Unlisted:              getBool(modelMap, "unlisted"),
		Managed:               managed,
		BenchyTrustRemoteCode: benchyTrustRemoteCode,
	}, true
}

func findModelGroup(modelID string, groups map[string]any) string {
	for groupName, raw := range groups {
		group, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		for _, member := range groupMembers(group) {
			if member == modelID {
				return groupName
			}
		}
	}
	return ""
}

func intFromAny(v any) int {
	switch t := v.(type) {
	case int:
		return t
	case int32:
		return int(t)
	case int64:
		return int(t)
	case float64:
		return int(t)
	case string:
		i, _ := strconv.Atoi(strings.TrimSpace(t))
		return i
	default:
		return 0
	}
}

func quoteForCommand(s string) string {
	if strings.ContainsAny(s, " \t\"") {
		return strconv.Quote(s)
	}
	return s
}

func tailString(s string, max int) string {
	if max <= 0 || len(s) <= max {
		return s
	}
	return "...(truncated)\n" + s[len(s)-max:]
}

// NVIDIA Image Functions

func loadNVIDIASourceImage(backendDir string) string {
	overrideFile := filepath.Join(backendDir, nvidiaSourceImageOverrideFile)
	data, err := os.ReadFile(overrideFile)
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(data))
}

func persistNVIDIASourceImage(backendDir, image string) error {
	overrideFile := filepath.Join(backendDir, nvidiaSourceImageOverrideFile)
	return os.WriteFile(overrideFile, []byte(strings.TrimSpace(image)), 0644)
}

// isNVIDIANGCImage checks if an image reference is from NVIDIA NGC Catalog
func isNVIDIANGCImage(image string) bool {
	image = strings.TrimSpace(image)
	// Valid NVIDIA NGC images start with "nvcr.io/nvidia/"
	return strings.HasPrefix(image, "nvcr.io/nvidia/")
}

func readDefaultNVIDIASourceImage(backendDir string) string {
	image := loadNVIDIASourceImage(backendDir)
	if image != "" && isNVIDIANGCImage(image) {
		return image
	}
	return defaultNVIDIASourceImage
}

func resolveNVIDIASourceImage(backendDir, requested string) string {
	if requested != "" {
		return requested
	}
	return readDefaultNVIDIASourceImage(backendDir)
}

func fetchNVIDIAReleaseTags(ctx context.Context) ([]string, error) {
	// Known NVIDIA vLLM image versions from NGC Catalog
	// These are the most commonly used versions
	knownVersions := []string{
		"26.01-py3",
		"25.03-py3",
		"25.02-py3",
		"25.01-py3",
		"24.12-py3",
		"24.11-py3",
		"24.10-py3",
		"24.09-py3",
		"24.08-py3",
		"24.07-py3",
		"24.06-py3",
		"24.05-py3",
	}

	tags := make([]string, 0, len(knownVersions))
	for _, version := range knownVersions {
		imageRef := fmt.Sprintf("nvcr.io/nvidia/vllm:%s", version)
		tags = append(tags, imageRef)
	}

	return tags, nil
}

func latestNVIDIATag(tags []string) string {
	if len(tags) == 0 {
		return ""
	}

	// Tags are now full image references like "nvcr.io/nvidia/vllm:26.01-py3"
	// Extract versions and sort them
	type version struct {
		imageRef string
		version  string
		major    int
		minor    int
		patch    int
	}

	versions := make([]version, 0, len(tags))
	for _, imageRef := range tags {
		ver := extractNVIDIAVersion(imageRef)
		if ver != "" {
			// Parse version like "26.01-py3" or "26.01"
			var major, minor, patch int
			var numFields int
			n, _ := fmt.Sscanf(ver, "%d.%d.%d-%*s%n", &major, &minor, &patch, &numFields)
			if n < 3 {
				n, _ = fmt.Sscanf(ver, "%d.%d-%*s%n", &major, &minor, &numFields)
			}
			if n >= 2 {
				versions = append(versions, version{
					imageRef: imageRef,
					version:  ver,
					major:   major,
					minor:   minor,
					patch:   patch,
				})
			}
		}
	}

	if len(versions) == 0 {
		return tags[0]
	}

	// Sort by major.minor.patch descending
	sort.Slice(versions, func(i, j int) bool {
		if versions[i].major != versions[j].major {
			return versions[i].major > versions[j].major
		}
		if versions[i].minor != versions[j].minor {
			return versions[i].minor > versions[j].minor
		}
		return versions[i].patch > versions[j].patch
	})

	return versions[0].imageRef
}

func extractNVIDIAVersion(imageRef string) string {
	// Extract version from image ref like "nvcr.io/nvidia/vllm:26.01-py3"
	parts := strings.Split(imageRef, ":")
	if len(parts) >= 2 {
		return parts[1]
	}
	return ""
}

func topNVIDIATags(tags []string, limit int) []string {
	if len(tags) <= limit {
		return tags
	}

	// Tags are full image references, sort them by version
	type version struct {
		imageRef string
		version  string
		major    int
		minor    int
		patch    int
	}

	versions := make([]version, 0, len(tags))
	for _, imageRef := range tags {
		ver := extractNVIDIAVersion(imageRef)
		if ver != "" {
			var major, minor, patch int
			var numFields int
			n, _ := fmt.Sscanf(ver, "%d.%d.%d-%*s%n", &major, &minor, &patch, &numFields)
			if n < 3 {
				n, _ = fmt.Sscanf(ver, "%d.%d-%*s%n", &major, &minor, &numFields)
			}
			if n >= 2 {
				versions = append(versions, version{
					imageRef: imageRef,
					version:  ver,
					major:   major,
					minor:   minor,
					patch:   patch,
				})
			}
		}
	}

	// Sort by major.minor.patch descending
	sort.Slice(versions, func(i, j int) bool {
		if versions[i].major != versions[j].major {
			return versions[i].major > versions[j].major
		}
		if versions[i].minor != versions[j].minor {
			return versions[i].minor > versions[j].minor
		}
		return versions[i].patch > versions[j].patch
	})

	// Return top limit image refs
	result := make([]string, 0, limit)
	for i := 0; i < limit && i < len(versions); i++ {
		result = append(result, versions[i].imageRef)
	}
	return result
}

func buildNVIDIAImageState(backendDir string) *RecipeBackendNVIDIAImage {
	// Default is always the official NVIDIA NGC image
	defaultImage := defaultNVIDIASourceImage

	// Try to load selected image from file, but validate it's an NVIDIA image
	selectedImage := loadNVIDIASourceImage(backendDir)
	if selectedImage == "" || !isNVIDIANGCImage(selectedImage) {
		// If file doesn't exist or contains invalid image, use default
		selectedImage = defaultImage
	}

	state := &RecipeBackendNVIDIAImage{
		Selected: selectedImage,
		Default:  defaultImage,
	}
	state.Available = appendUniqueString(state.Available, selectedImage)
	state.Available = appendUniqueString(state.Available, defaultImage)

	ctx, cancel := context.WithTimeout(context.Background(), 8*time.Second)
	defer cancel()

	tags, err := fetchNVIDIAReleaseTags(ctx)
	if err != nil {
		state.Warning = fmt.Sprintf("No se pudieron consultar tags de NGC: %v", err)
		return state
	}

	latestImage := latestNVIDIATag(tags)
	if latestImage != "" {
		state.Latest = latestImage
		state.Available = appendUniqueString(state.Available, latestImage)

		selectedVersion := extractNVIDIAVersion(selectedImage)
		latestVersion := extractNVIDIAVersion(latestImage)
		if selectedVersion != "" && latestVersion != "" && selectedVersion != latestVersion {
			state.UpdateAvailable = true
		}
	}

	for _, tag := range topNVIDIATags(tags, 12) {
		state.Available = appendUniqueString(state.Available, tag)
	}
	return state
}
