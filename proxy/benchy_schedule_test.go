package proxy

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestParseBenchyStartAtRFC3339(t *testing.T) {
	raw := "2026-02-17T12:34:56Z"
	got, err := parseBenchyStartAt(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got == nil {
		t.Fatalf("expected parsed time, got nil")
	}
	if got.Format(time.RFC3339) != raw {
		t.Fatalf("unexpected parse result\nwant: %q\ngot:  %q", raw, got.Format(time.RFC3339))
	}
}

func TestParseBenchyStartAtInvalid(t *testing.T) {
	got, err := parseBenchyStartAt("not-a-date")
	if err == nil {
		t.Fatalf("expected parse error, got nil (value=%v)", got)
	}
}

func TestWaitForScheduledBenchyStartNilStart(t *testing.T) {
	pm := &ProxyManager{
		benchyJobs: map[string]*BenchyJob{
			"job-1": {ID: "job-1", Status: benchyStatusScheduled},
		},
	}

	if err := pm.waitForScheduledBenchyStart(context.Background(), "job-1", nil); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if pm.benchyJobs["job-1"].Status != benchyStatusRunning {
		t.Fatalf("expected status %q, got %q", benchyStatusRunning, pm.benchyJobs["job-1"].Status)
	}
}

func TestWaitForScheduledBenchyStartCanceled(t *testing.T) {
	pm := &ProxyManager{
		benchyJobs: map[string]*BenchyJob{
			"job-2": {ID: "job-2", Status: benchyStatusScheduled},
		},
	}
	startAt := time.Now().Add(500 * time.Millisecond)

	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()

	err := pm.waitForScheduledBenchyStart(ctx, "job-2", &startAt)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got: %v", err)
	}
}
