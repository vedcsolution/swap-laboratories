<script lang="ts">
  import { onMount } from "svelte";
  import { getRecipeBackendState, runRecipeBackendAction, setRecipeBackend } from "../stores/api";
  import type { RecipeBackendAction, RecipeBackendState } from "../lib/types";
  import { collapseHomePath } from "../lib/pathDisplay";

  let loading = $state(true);
  let refreshing = $state(false);
  let saving = $state(false);
  let error = $state<string | null>(null);
  let notice = $state<string | null>(null);
  let state = $state<RecipeBackendState | null>(null);
  let selected = $state("");
  let customPath = $state("");
  let useCustom = $state(false);
  let actionRunning = $state<string>("");
  let actionCommand = $state("");
  let actionOutput = $state("");
  let selectedTrtllmImage = $state("");
  let selectedNvidiaImage = $state("");
  let refreshController: AbortController | null = null;

  function sourceLabel(source: RecipeBackendState["backendSource"]): string {
    if (source === "override") return "override (UI)";
    if (source === "env") return "env";
    return "default";
  }

  function backendKindLabel(kind: RecipeBackendState["backendKind"], vendor?: string): string {
    const v = (vendor || "").trim();
    if (!v) return kind;
    return `${kind} (${v})`;
  }

  function isNvidiaBackendOption(path: string): boolean {
    return path.toLowerCase().includes("spark-trtllm-docker") || path.toLowerCase().includes("spark-vllm-docker-nvidia");
  }

  function syncSelectionFromState(next: RecipeBackendState): void {
    if (next.options.includes(next.backendDir)) {
      selected = next.backendDir;
      customPath = "";
      useCustom = false;
    } else {
      selected = "";
      customPath = next.backendDir;
      useCustom = true;
    }

    if (next.trtllmImage?.selected) {
      selectedTrtllmImage = next.trtllmImage.selected;
    } else {
      selectedTrtllmImage = "";
    }

    if (next.nvidiaImage?.selected) {
      selectedNvidiaImage = next.nvidiaImage.selected;
    } else {
      selectedNvidiaImage = "";
    }
  }

  function trtllmImageOptions(next: RecipeBackendState | null): string[] {
    if (!next?.trtllmImage) return [];
    const out: string[] = [];
    const push = (v?: string) => {
      const value = (v || "").trim();
      if (!value || out.includes(value)) return;
      out.push(value);
    };

    push(next.trtllmImage.selected);
    push(next.trtllmImage.default);
    push(next.trtllmImage.latest);
    for (const img of next.trtllmImage.available || []) {
      push(img);
    }
    return out;
  }

  function nvidiaImageOptions(next: RecipeBackendState | null): string[] {
    if (!next?.nvidiaImage) return [];
    const out: string[] = [];
    const push = (v?: string) => {
      const value = (v || "").trim();
      if (!value || out.includes(value)) return;
      out.push(value);
    };

    push(next.nvidiaImage.selected);
    push(next.nvidiaImage.default);
    push(next.nvidiaImage.latest);
    for (const img of next.nvidiaImage.available || []) {
      push(img);
    }
    return out;
  }

  function runningLabel(action: string): string {
    switch (action) {
      case "git_pull":
        return "Running git pull...";
      case "git_pull_rebase":
        return "Running rebase pull...";
      case "build_vllm":
        return "Building vLLM...";
      case "build_mxfp4":
        return "Building MXFP4...";
      case "build_trtllm_image":
        return "Building TRT-LLM image...";
      case "update_trtllm_image":
        return "Updating TRT-LLM image...";
      case "pull_nvidia_image":
        return "Pulling NVIDIA image...";
      case "update_nvidia_image":
        return "Updating NVIDIA image...";
      default:
        return "Running...";
    }
  }

  async function refresh(): Promise<void> {
    refreshController?.abort();
    const controller = new AbortController();
    refreshController = controller;
    const timeout = setTimeout(() => controller.abort(), 15000);

    refreshing = true;
    error = null;
    notice = null;
    if (!state) loading = true;
    try {
      const next = await getRecipeBackendState(controller.signal);
      state = next;
      syncSelectionFromState(next);
    } catch (e) {
      if (controller.signal.aborted) {
        error = "Timeout consultando backend. Pulsa Refresh para reintentar.";
      } else {
        error = e instanceof Error ? e.message : String(e);
      }
    } finally {
      clearTimeout(timeout);
      if (refreshController === controller) {
        refreshController = null;
      }
      refreshing = false;
      loading = false;
    }
  }

  async function applySelection(): Promise<void> {
    if (saving) return;
    const backendDir = useCustom ? customPath.trim() : selected.trim();
    if (!backendDir) {
      error = "Selecciona un backend o introduce una ruta.";
      return;
    }

    saving = true;
    error = null;
    notice = null;
    actionCommand = "";
    actionOutput = "";
    try {
      const next = await setRecipeBackend(backendDir);
      state = next;
      syncSelectionFromState(next);
      notice = "Backend actualizado correctamente.";
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      saving = false;
    }
  }

  function formatDuration(ms: number): string {
    if (ms < 1000) return `${ms} ms`;
    return `${(ms / 1000).toFixed(1)} s`;
  }

  async function runAction(action: string, label: string): Promise<void> {
    if (actionRunning) return;
    actionRunning = action;
    error = null;
    notice = null;
    actionCommand = "";
    actionOutput = "";

    try {
      const opts =
        (action === "build_trtllm_image" || action === "update_trtllm_image") && selectedTrtllmImage.trim()
          ? { sourceImage: selectedTrtllmImage.trim() }
          : (action === "pull_nvidia_image" || action === "update_nvidia_image") && selectedNvidiaImage.trim()
          ? { sourceImage: selectedNvidiaImage.trim() }
          : undefined;
      const result = await runRecipeBackendAction(action as RecipeBackendAction, opts);
      actionCommand = result.command || "";
      actionOutput = result.output || "";
      notice = `${label} completado en ${formatDuration(result.durationMs || 0)}.`;
      if (
        action === "git_pull" ||
        action === "git_pull_rebase" ||
        action === "build_trtllm_image" ||
        action === "update_trtllm_image" ||
        action === "pull_nvidia_image" ||
        action === "update_nvidia_image"
      ) {
        await refresh();
      }
    } catch (e) {
      error = e instanceof Error ? e.message : String(e);
    } finally {
      actionRunning = "";
    }
  }

  onMount(() => {
    void refresh();
    return () => {
      refreshController?.abort();
    };
  });
</script>

<div class="h-full flex flex-col gap-2">
  <div class="card shrink-0">
    <div class="flex items-center justify-between gap-2">
      <h2 class="pb-0">Backend</h2>
      <button class="btn btn--sm" onclick={refresh} disabled={refreshing || saving}>
        {refreshing ? "Refreshing..." : "Refresh"}
      </button>
    </div>

    {#if state}
      <div class="mt-2 text-sm text-txtsecondary break-all">
        Actual:
        <span class="font-mono text-txtmain" title={state.backendDir}>{collapseHomePath(state.backendDir)}</span>
      </div>
      <div class="text-xs text-txtsecondary">Fuente: {sourceLabel(state.backendSource)}</div>
      <div class="text-xs text-txtsecondary">
        Tipo backend:
        <span class="font-mono text-txtmain">{backendKindLabel(state.backendKind, state.backendVendor)}</span>
        {#if state.repoUrl}
          | repo: <span class="font-mono text-txtmain">{state.repoUrl}</span>
        {/if}
      </div>
    {/if}

    {#if error}
      <div class="mt-2 p-2 border border-error/40 bg-error/10 rounded text-sm text-error break-words">{error}</div>
    {/if}
    {#if notice}
      <div class="mt-2 p-2 border border-green-400/30 bg-green-600/10 rounded text-sm text-green-300 break-words">{notice}</div>
    {/if}
  </div>

  <div class="card flex-1 min-h-0 overflow-auto">
    {#if loading}
      <div class="text-sm text-txtsecondary">Cargando opciones de backend...</div>
    {:else if state}
      <div class="text-sm text-txtsecondary mb-2">
        Selecciona qué backend usar para recetas y operaciones de cluster.
      </div>

      <div class="space-y-2">
        {#each state.options as option}
          <label class="flex items-center gap-2 text-sm">
            <input
              type="radio"
              name="backend-option"
              checked={!useCustom && selected === option}
              onchange={() => {
                useCustom = false;
                selected = option;
              }}
            />
            <span class="font-mono break-all" title={option}>{collapseHomePath(option)}</span>
            {#if isNvidiaBackendOption(option)}
              <span class="text-[11px] px-1.5 py-0.5 rounded border border-cyan-400/40 text-cyan-300">nvidia</span>
            {/if}
          </label>
        {/each}

        <label class="flex items-center gap-2 text-sm pt-1">
          <input
            type="radio"
            name="backend-option"
            checked={useCustom}
            onchange={() => {
              useCustom = true;
            }}
          />
          <span>Ruta personalizada</span>
        </label>
        <input
          class="input w-full font-mono text-sm"
          placeholder="~/spark-vllm-docker"
          bind:value={customPath}
          onfocus={() => {
            useCustom = true;
          }}
        />
      </div>

      <div class="mt-3">
        <button class="btn btn--sm" onclick={applySelection} disabled={saving || refreshing}>
          {saving ? "Applying..." : "Apply Backend"}
        </button>
      </div>

      {#if state.backendKind === "trtllm" && state.trtllmImage}
        <div class="mt-4 p-3 border border-card-border rounded bg-background/40 space-y-2">
          <div class="text-sm text-txtsecondary">TRT-LLM source image (NVIDIA)</div>
          {#if state.deploymentGuideUrl}
            <div class="text-xs text-txtsecondary break-all">
              Guía técnica:
              <a class="underline text-cyan-300" href={state.deploymentGuideUrl} target="_blank" rel="noreferrer">{state.deploymentGuideUrl}</a>
            </div>
          {/if}
          <select class="w-full px-2 py-1 rounded border border-card-border bg-background font-mono text-sm" bind:value={selectedTrtllmImage}>
            {#each trtllmImageOptions(state) as image}
              <option value={image}>{image}</option>
            {/each}
          </select>
          <input
            class="input w-full font-mono text-sm"
            bind:value={selectedTrtllmImage}
            placeholder="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc3"
          />
          <div class="text-xs text-txtsecondary break-all">default: <span class="font-mono">{state.trtllmImage.default}</span></div>
          {#if state.trtllmImage.latest}
            <div class="text-xs text-txtsecondary break-all">
              latest: <span class="font-mono">{state.trtllmImage.latest}</span>
              {#if state.trtllmImage.updateAvailable}
                <span class="ml-2 text-amber-300 font-semibold">update available</span>
              {:else}
                <span class="ml-2 text-green-300">up-to-date</span>
              {/if}
            </div>
          {/if}
          {#if state.trtllmImage.warning}
            <div class="p-2 border border-amber-500/30 bg-amber-500/10 rounded text-xs text-amber-300 break-words">
              {state.trtllmImage.warning}
            </div>
          {/if}
          <div class="text-xs text-txtsecondary">Se guarda como preferencia de imagen para despliegues TRT-LLM.</div>
        </div>
      {/if}

      {#if state.backendKind === "nvidia" && state.nvidiaImage}
        <div class="mt-4 p-3 border border-card-border rounded bg-background/40 space-y-2">
          <div class="text-sm text-txtsecondary">NVIDIA vLLM image (vllm-openai)</div>
          {#if state.deploymentGuideUrl}
            <div class="text-xs text-txtsecondary break-all">
              Guía técnica:
              <a class="underline text-cyan-300" href={state.deploymentGuideUrl} target="_blank" rel="noreferrer">{state.deploymentGuideUrl}</a>
            </div>
          {/if}
          <select class="w-full px-2 py-1 rounded border border-card-border bg-background font-mono text-sm" bind:value={selectedNvidiaImage}>
            {#each nvidiaImageOptions(state) as image}
              <option value={image}>{image}</option>
            {/each}
          </select>
          <input
            class="input w-full font-mono text-sm"
            bind:value={selectedNvidiaImage}
            placeholder="vllm/vllm-openai:v0.6.6.post1"
          />
          <div class="text-xs text-txtsecondary break-all">default: <span class="font-mono">{state.nvidiaImage.default}</span></div>
          {#if state.nvidiaImage.latest}
            <div class="text-xs text-txtsecondary break-all">
              latest: <span class="font-mono">{state.nvidiaImage.latest}</span>
              {#if state.nvidiaImage.updateAvailable}
                <span class="ml-2 text-amber-300 font-semibold">update available</span>
              {:else}
                <span class="ml-2 text-green-300">up-to-date</span>
              {/if}
            </div>
          {/if}
          {#if state.nvidiaImage.warning}
            <div class="p-2 border border-amber-500/30 bg-amber-500/10 rounded text-xs text-amber-300 break-words">
              {state.nvidiaImage.warning}
            </div>
          {/if}
          <div class="text-xs text-txtsecondary">Se guarda como preferencia de imagen para despliegues NVIDIA vLLM.</div>
        </div>
      {/if}

      <div class="mt-4 pt-3 border-t border-card-border">
        <div class="text-sm text-txtsecondary mb-2">Backend actions</div>

        {#if state.actions.length === 0}
          <div class="text-xs text-txtsecondary">No hay acciones disponibles para este backend.</div>
        {:else}
          <div class="flex flex-wrap gap-2">
            {#each state.actions as info (info.action + info.label)}
              <button
                class="btn btn--sm"
                onclick={() => runAction(info.action, info.label)}
                disabled={!!actionRunning || saving || refreshing}
                title={info.commandHint || info.label}
              >
                {actionRunning === info.action ? runningLabel(info.action) : info.label}
              </button>
            {/each}
          </div>
        {/if}

        {#if actionCommand}
          <div class="mt-2 text-xs text-txtsecondary break-all">
            Command: <span class="font-mono">{actionCommand}</span>
          </div>
        {/if}
        {#if actionOutput}
          <pre class="mt-2 p-2 border border-card-border rounded bg-background/60 text-xs font-mono whitespace-pre-wrap break-all max-h-72 overflow-auto">{actionOutput}</pre>
        {/if}
      </div>
    {:else}
      <div class="text-sm text-txtsecondary">No se pudo cargar el estado del backend.</div>
    {/if}
  </div>
</div>
