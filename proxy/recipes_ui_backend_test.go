package proxy

import (
	"os"
	"path/filepath"
	"testing"
)

func TestDetectRecipeBackendKind(t *testing.T) {
	tests := []struct {
		path string
		want string
	}{
		{path: "/home/u/spark-vllm-docker", want: "vllm"},
		{path: "/home/u/spark-sqlang-docker", want: "sqlang"},
		{path: "/home/u/spark-trtllm-docker", want: "trtllm"},
		{path: "/opt/custom-backend", want: "custom"},
	}

	for _, tc := range tests {
		got := detectRecipeBackendKind(tc.path)
		if got != tc.want {
			t.Fatalf("detectRecipeBackendKind(%q) = %q, want %q", tc.path, got, tc.want)
		}
	}
}

func TestLatestTRTLLMTag(t *testing.T) {
	tags := []string{"1.3.0rc3", "1.2.5", "1.3.0", "1.3.0rc4", "1.4.0rc1", "latest", "1.4.0"}
	got := latestTRTLLMTag(tags)
	want := "1.4.0"
	if got != want {
		t.Fatalf("latestTRTLLMTag() = %q, want %q", got, want)
	}
}

func TestCompareTRTLLMTagVersion(t *testing.T) {
	a, ok := parseTRTLLMTagVersion("1.3.0rc3")
	if !ok {
		t.Fatalf("failed to parse a")
	}
	b, ok := parseTRTLLMTagVersion("1.3.0")
	if !ok {
		t.Fatalf("failed to parse b")
	}
	if compareTRTLLMTagVersion(a, b) >= 0 {
		t.Fatalf("expected rc version to be lower than stable")
	}

	c, ok := parseTRTLLMTagVersion("1.3.1")
	if !ok {
		t.Fatalf("failed to parse c")
	}
	if compareTRTLLMTagVersion(b, c) >= 0 {
		t.Fatalf("expected 1.3.0 < 1.3.1")
	}
}

func TestResolveTRTLLMSourceImagePrefersOverrideFile(t *testing.T) {
	dir := t.TempDir()
	overridePath := filepath.Join(dir, trtllmSourceImageOverrideFile)
	overrideValue := "nvcr.io/nvidia/tensorrt-llm/release:1.4.0"
	if err := os.WriteFile(overridePath, []byte(overrideValue+"\n"), 0o644); err != nil {
		t.Fatalf("write override: %v", err)
	}

	got := resolveTRTLLMSourceImage(dir, "")
	if got != overrideValue {
		t.Fatalf("resolveTRTLLMSourceImage() = %q, want %q", got, overrideValue)
	}
}

func TestBackendScopedConfigPath(t *testing.T) {
	cfg := "/tmp/config.yaml"
	if got, want := backendScopedConfigPath(cfg, "vllm"), "/tmp/config.vllm.yaml"; got != want {
		t.Fatalf("backendScopedConfigPath(vllm) = %q, want %q", got, want)
	}
	if got, want := backendScopedConfigPath(cfg, "trtllm"), "/tmp/config.trtllm.yaml"; got != want {
		t.Fatalf("backendScopedConfigPath(trtllm) = %q, want %q", got, want)
	}
	if got, want := backendScopedConfigPath(cfg, "unknown"), "/tmp/config.custom.yaml"; got != want {
		t.Fatalf("backendScopedConfigPath(custom) = %q, want %q", got, want)
	}
}

func TestSwitchRecipeBackendConfigPersistsAndRestores(t *testing.T) {
	dir := t.TempDir()
	activePath := filepath.Join(dir, "config.yaml")
	vllmBody := `macros:
  recipe_runner: /vllm/run-recipe.sh
`
	trtBody := `macros:
  recipe_runner: /trt/run-recipe.sh
`

	if err := os.WriteFile(activePath, []byte(vllmBody), 0o644); err != nil {
		t.Fatalf("write active config: %v", err)
	}
	trtPath := backendScopedConfigPath(activePath, "trtllm")
	if err := os.WriteFile(trtPath, []byte(trtBody), 0o644); err != nil {
		t.Fatalf("write trt config: %v", err)
	}

	pm := &ProxyManager{configPath: activePath}
	if err := pm.switchRecipeBackendConfig("vllm", "trtllm"); err != nil {
		t.Fatalf("switchRecipeBackendConfig error: %v", err)
	}

	activeGot, err := os.ReadFile(activePath)
	if err != nil {
		t.Fatalf("read active config: %v", err)
	}
	if string(activeGot) != trtBody {
		t.Fatalf("active config mismatch\nwant:\n%s\ngot:\n%s", trtBody, string(activeGot))
	}

	vllmPath := backendScopedConfigPath(activePath, "vllm")
	vllmGot, err := os.ReadFile(vllmPath)
	if err != nil {
		t.Fatalf("read vllm scoped config: %v", err)
	}
	if string(vllmGot) != vllmBody {
		t.Fatalf("vllm scoped config mismatch\nwant:\n%s\ngot:\n%s", vllmBody, string(vllmGot))
	}
}
