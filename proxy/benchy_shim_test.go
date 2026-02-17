package proxy

import (
	"os"
	"path/filepath"
	"testing"
)

func createSiteCustomize(t *testing.T, dir string) {
	t.Helper()
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatalf("mkdir %s: %v", dir, err)
	}
	path := filepath.Join(dir, "sitecustomize.py")
	if err := os.WriteFile(path, []byte("# shim\n"), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func TestBenchyPythonShimDirEnvOverride(t *testing.T) {
	root := t.TempDir()
	shimDir := filepath.Join(root, "custom-shim")
	createSiteCustomize(t, shimDir)

	t.Setenv(benchyEnvPyShimDir, shimDir)
	pm := &ProxyManager{configPath: "/does/not/matter/config.yaml"}

	got := pm.benchyPythonShimDir()
	want := filepath.Clean(shimDir)
	if got != want {
		t.Fatalf("unexpected shim dir\nwant: %q\ngot:  %q", want, got)
	}
}

func TestBenchyPythonShimDirRelativeConfigPath(t *testing.T) {
	t.Setenv(benchyEnvPyShimDir, "")

	root := t.TempDir()
	shimDir := filepath.Join(root, "proxy", "pyshim")
	createSiteCustomize(t, shimDir)
	cfgPath := filepath.Join(root, "config.yaml")
	if err := os.WriteFile(cfgPath, []byte("version: 1\n"), 0o644); err != nil {
		t.Fatalf("write %s: %v", cfgPath, err)
	}

	prevWD, err := os.Getwd()
	if err != nil {
		t.Fatalf("getwd: %v", err)
	}
	if err := os.Chdir(root); err != nil {
		t.Fatalf("chdir %s: %v", root, err)
	}
	t.Cleanup(func() {
		_ = os.Chdir(prevWD)
	})

	pm := &ProxyManager{configPath: "./config.yaml"}
	got := pm.benchyPythonShimDir()
	want := filepath.Clean(shimDir)
	if got != want {
		t.Fatalf("unexpected shim dir\nwant: %q\ngot:  %q", want, got)
	}
}
