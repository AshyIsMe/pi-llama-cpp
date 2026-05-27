import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { spawn } from "node:child_process";
import type { ChildProcess } from "node:child_process";
import { closeSync, constants, openSync, readFileSync, writeSync } from "node:fs";
import { access, appendFile, chmod, mkdir, open as openFile, readFile, readdir, realpath, rename, rm, stat, unlink, writeFile } from "node:fs/promises";
import { arch, homedir, platform, totalmem } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const EXTENSION_DIR = dirname(fileURLToPath(import.meta.url));
const PROVIDER_ID = "llama-cpp";
const MODEL_CATALOG = [
	{ id: "qwen3.6-27b", name: "Qwen 3.6 27B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.6-27B-GGUF", prefix: "LLAMA_CPP_QWEN36_27B", legacyPrefix: "LLAMA_CPP_QWEN36" },
	{ id: "qwen3.6-35b-a3b", name: "Qwen 3.6 35B-A3B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.6-35B-A3B-GGUF", prefix: "LLAMA_CPP_QWEN36_35B_A3B" },
	{ id: "qwen3.6-27b-mtp", name: "Qwen 3.6 27B MTP (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.6-27B-MTP-GGUF", prefix: "LLAMA_CPP_QWEN36_27B_MTP" },
	{ id: "qwen3.6-35b-a3b-mtp", name: "Qwen 3.6 35B-A3B MTP (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.6-35B-A3B-MTP-GGUF", prefix: "LLAMA_CPP_QWEN36_35B_A3B_MTP" },
	{ id: "qwen3.5-0.8b", name: "Qwen 3.5 0.8B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-0.8B-GGUF", prefix: "LLAMA_CPP_QWEN35_0_8B" },
	{ id: "qwen3.5-2b", name: "Qwen 3.5 2B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-2B-GGUF", prefix: "LLAMA_CPP_QWEN35_2B" },
	{ id: "qwen3.5-4b", name: "Qwen 3.5 4B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-4B-GGUF", prefix: "LLAMA_CPP_QWEN35_4B" },
	{ id: "qwen3.5-9b", name: "Qwen 3.5 9B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-9B-GGUF", prefix: "LLAMA_CPP_QWEN35_9B" },
	{ id: "qwen3.5-27b", name: "Qwen 3.5 27B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-27B-GGUF", prefix: "LLAMA_CPP_QWEN35_27B" },
	{ id: "qwen3.5-35b-a3b", name: "Qwen 3.5 35B-A3B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-35B-A3B-GGUF", prefix: "LLAMA_CPP_QWEN35_35B_A3B", legacyPrefix: "LLAMA_CPP_QWEN35" },
	{ id: "qwen3.5-122b-a10b", name: "Qwen 3.5 122B-A10B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-122B-A10B-GGUF", prefix: "LLAMA_CPP_QWEN35_122B_A10B" },
	{ id: "qwen3.5-397b-a17b", name: "Qwen 3.5 397B-A17B (Unsloth GGUF, llama.cpp)", repo: "unsloth/Qwen3.5-397B-A17B-GGUF", prefix: "LLAMA_CPP_QWEN35_397B_A17B" },
] as const;
const MANAGED_BY = "pi-llama-cpp-provider";

const ROOT_DIR = join(homedir(), ".pi", "llama-cpp");
const SETTINGS_FILE = join(ROOT_DIR, "settings.json");
const RUNTIME_DIR = join(ROOT_DIR, "runtime");
const MODELS_DIR = join(ROOT_DIR, "models");
const CLIENT_DIR = join(ROOT_DIR, "clients");
const LOCK_DIR = join(ROOT_DIR, "lock");
const STATE_FILE = join(ROOT_DIR, "server.json");
const LOG_FILE = join(ROOT_DIR, "log");
const LEASE_FILE = join(CLIENT_DIR, `${process.pid}.json`);
const WATCHDOG_SCRIPT = join(EXTENSION_DIR, "llama-cpp-watchdog.sh");

const BASE_URL = "http://127.0.0.1:8080";
const API_BASE_URL = `${BASE_URL}/v1`;
const HEARTBEAT_MS = 10_000;
const LEASE_TTL_MS = 45_000;
const LOCK_TIMEOUT_MS = 30_000;
const STARTUP_LOCK_TIMEOUT_MS = 24 * 60 * 60_000;
const HTTP_CHECK_TIMEOUT_MS = 1500;
const SHUTDOWN_GRACE_MS = 60_000;
const PROGRESS_NOTIFY_MS = 750;

type ProviderProtocol = "openai-completions" | "openai-responses";
type Backend = "cpu" | "vulkan" | "cuda" | "rocm" | "metal";
type ModelKey = (typeof MODEL_CATALOG)[number]["id"];
type Settings = Record<string, unknown>;
type StatusCallback = (message: string | undefined) => void;
type RunOptions = { onStatus?: StatusCallback; progressPrefix?: string };

type ServerState = {
	managedBy: string;
	pid: number;
	baseUrl: string;
	port: number;
	cwd: string;
	binary: string;
	args: string[];
	backend: Backend;
	modelId: ModelKey;
	modelPath: string;
	startedAt: number;
	startedAtIso: string;
	stopping?: boolean;
};

type Lease = { managedBy: string; usesLlamaCpp: true; pid: number; processStart: string; cwd: string; startedAt: number; updatedAt: number; updatedAtIso: string };

type ModelSpec = { id: ModelKey; name: string; repo: string; file?: string; quant: string; contextWindow: number; maxTokens: number };

function readSettingsSync(): Settings {
	try {
		const data = JSON.parse(readFileSync(SETTINGS_FILE, "utf8"));
		if (data && typeof data === "object" && !Array.isArray(data)) return data as Settings;
		throw new Error("settings root must be an object");
	} catch (error: any) {
		if (error?.code === "ENOENT") return {};
		throw new Error(`Failed to read ${SETTINGS_FILE}: ${describeError(error)}`);
	}
}
const SETTINGS = readSettingsSync();
const READY_TIMEOUT_MS = configNumber("LLAMA_CPP_READY_TIMEOUT_MS", 10 * 60_000);

function keyForEnv(envName: string): string { return envName.replace(/^LLAMA_CPP_/, "").toLowerCase().replace(/_([a-z])/g, (_m, c: string) => c.toUpperCase()); }
function settingValue(envName: string): unknown {
	if (process.env[envName] !== undefined) return process.env[envName];
	const snake = envName.replace(/^LLAMA_CPP_/, "").toLowerCase();
	for (const key of [envName, keyForEnv(envName), envName.toLowerCase(), snake]) if (Object.prototype.hasOwnProperty.call(SETTINGS, key)) return SETTINGS[key];
	return undefined;
}
function configString(envName: string, def?: string): string | undefined {
	const value = settingValue(envName);
	if (value === undefined || value === null || value === "") return def;
	if (["string", "number", "boolean"].includes(typeof value)) return String(value);
	throw new Error(`${envName} must be a string in the environment or ${SETTINGS_FILE}`);
}
function configNumber(envName: string, def: number): number {
	const value = settingValue(envName);
	if (value === undefined || value === null || value === "") return def;
	const n = typeof value === "number" ? value : typeof value === "string" ? Number(value) : NaN;
	if (!Number.isFinite(n)) throw new Error(`${envName} must be a finite number`);
	return n;
}
function selectedProtocol(): ProviderProtocol {
	const raw = configString("LLAMA_CPP_PROTOCOL", "openai")!.toLowerCase();
	if (["openai", "chat", "chat-completions", "openai-completions"].includes(raw)) return "openai-completions";
	if (["responses", "openai-responses"].includes(raw)) return "openai-responses";
	throw new Error(`Invalid LLAMA_CPP_PROTOCOL=${raw}`);
}

const PROVIDER_API = selectedProtocol();
let heartbeat: ReturnType<typeof setInterval> | undefined;
let startupPromise: Promise<void> | undefined;
let startupModelId: ModelKey | undefined;
let activeChild: ChildProcess | undefined;
let leaseStartedAt = Date.now();
let ownProcessStart: string | undefined;
let watchdogStarted = false;
let runtimeDisposed = false;
let shuttingDown = false;
let writeSeq = 0;
let selectedBackend: Backend | undefined;

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));
function describeError(error: unknown): string { return error instanceof Error ? error.message : String(error); }
function isPidAlive(pid: unknown): pid is number { if (typeof pid !== "number" || !Number.isFinite(pid) || pid <= 0) return false; try { process.kill(pid, 0); return true; } catch (e: any) { return e?.code === "EPERM"; } }
function shellQuote(value: string): string { return /^[A-Za-z0-9_./:=+@%-]+$/.test(value) ? value : `'${value.replace(/'/g, `'\\''`)}'`; }
function formatBytes(bytes: number): string { if (bytes < 1024) return `${bytes} B`; if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`; if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`; return `${(bytes / 1024 ** 3).toFixed(1)} GB`; }
async function appendLog(text: string): Promise<void> { await mkdir(ROOT_DIR, { recursive: true }); await appendFile(LOG_FILE, text, "utf8"); }
async function readJson<T>(file: string): Promise<T | undefined> { try { return JSON.parse(await readFile(file, "utf8")) as T; } catch { return undefined; } }
async function writeJsonAtomic(file: string, data: unknown): Promise<void> { await mkdir(dirname(file), { recursive: true }); const tmp = `${file}.${process.pid}.${Date.now()}.${++writeSeq}.tmp`; await writeFile(tmp, `${JSON.stringify(data, null, 2)}\n`); await rename(tmp, file); }
async function removeFile(file: string): Promise<void> { try { await unlink(file); } catch (e: any) { if (e?.code !== "ENOENT") throw e; } }
async function execCapture(command: string, args: string[], timeoutMs = 2500): Promise<string | undefined> {
	return new Promise((resolve) => {
		let stdout = "", stderr = "", settled = false; let child: ChildProcess;
		const done = (v?: string) => { if (settled) return; settled = true; clearTimeout(timer); resolve(v); };
		const timer = setTimeout(() => { try { child?.kill("SIGTERM"); } catch {} done(undefined); }, timeoutMs); timer.unref?.();
		try { child = spawn(command, args, { stdio: ["ignore", "pipe", "pipe"] }); } catch { done(undefined); return; }
		child.stdout?.setEncoding("utf8"); child.stderr?.setEncoding("utf8"); child.stdout?.on("data", (c) => stdout += c); child.stderr?.on("data", (c) => stderr += c);
		child.on("error", () => done(undefined)); child.on("close", (code) => done(code === 0 ? stdout : stdout || stderr || undefined));
	});
}
async function processArgs(pid: number) { return (await execCapture("ps", ["-p", String(pid), "-o", "args="], 2000))?.trim(); }
async function processStart(pid: number) { return (await execCapture("ps", ["-p", String(pid), "-o", "lstart="], 2000))?.trim() || undefined; }
async function getOwnProcessStart() { ownProcessStart ??= (await processStart(process.pid)) ?? "unknown"; return ownProcessStart; }
async function looksLikeServer(pid: number) { return !!(await processArgs(pid))?.match(/(^|[/\s])llama-server(\s|$)/); }

function modelCatalogEntry(id: ModelKey) { return MODEL_CATALOG.find((m) => m.id === id)!; }
function modelConfigString(prefix: string, legacyPrefix: string | undefined, suffix: string, def?: string): string | undefined {
	return configString(`${prefix}_${suffix}`, legacyPrefix ? configString(`${legacyPrefix}_${suffix}`, def) : def);
}
function modelConfigNumber(prefix: string, legacyPrefix: string | undefined, suffix: string, globalEnv: string, def: number): number {
	return configNumber(`${prefix}_${suffix}`, legacyPrefix ? configNumber(`${legacyPrefix}_${suffix}`, configNumber(globalEnv, def)) : configNumber(globalEnv, def));
}
function modelSpec(id: ModelKey): ModelSpec {
	const entry = modelCatalogEntry(id);
	const legacyPrefix = "legacyPrefix" in entry ? entry.legacyPrefix : undefined;
	const repo = modelConfigString(entry.prefix, legacyPrefix, "REPO", entry.repo)!;
	const file = modelConfigString(entry.prefix, legacyPrefix, "FILE");
	const quant = modelConfigString(entry.prefix, legacyPrefix, "QUANT", configString("LLAMA_CPP_MODEL_QUANT", "Q4_K_M"))!;
	return { id, name: entry.name, repo, file, quant, contextWindow: modelConfigNumber(entry.prefix, legacyPrefix, "CTX", "LLAMA_CPP_CTX", 131072), maxTokens: modelConfigNumber(entry.prefix, legacyPrefix, "MAX_TOKENS", "LLAMA_CPP_MAX_TOKENS", 32768) };
}
function allModels() { return MODEL_CATALOG.map((m) => modelSpec(m.id)); }
function modelKeyForModelId(id?: string): ModelKey | undefined { return MODEL_CATALOG.some((m) => m.id === id) ? id as ModelKey : undefined; }

function createProgressReporter(prefix: string, onStatus?: StatusCallback) {
	let latest: string | undefined, emitted: string | undefined, last = 0, buffer = "";
	const emit = (force = false) => { if (!onStatus || !latest || latest === emitted) return; const now = Date.now(); if (!force && now - last < PROGRESS_NOTIFY_MS) return; emitted = latest; last = now; onStatus(`${prefix}: ${latest}`); };
	const processLine = (line: string) => { const s = line.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "").replace(/\s+/g, " ").trim(); if (!s || /^% Total\b/.test(s)) return; latest = s.length > 180 ? `${s.slice(0, 179)}…` : s; emit(); };
	return { onChunk(chunk: Buffer | string) { const t = chunk.toString(); let start = 0; for (let i = 0; i < t.length; i++) if (t[i] === "\n" || t[i] === "\r") { processLine(buffer + t.slice(start, i)); buffer = ""; start = i + 1; } buffer += t.slice(start); if (buffer) processLine(buffer); if (buffer.length > 4096) buffer = ""; }, flush() { if (buffer) processLine(buffer); emit(true); } };
}
async function runLogged(command: string, args: string[], cwd: string, label: string, options: RunOptions = {}) {
	if (runtimeDisposed || shuttingDown) throw new Error(`${label} cancelled`);
	await appendLog(`\n[${new Date().toISOString()}] ${label}\n$ ${[command, ...args].map(shellQuote).join(" ")}\n`);
	const logFd = openSync(LOG_FILE, "a"); const progress = options.progressPrefix ? createProgressReporter(options.progressPrefix, options.onStatus) : undefined; let closed = false;
	const writeChunk = (c: Buffer | string) => { if (!closed) try { writeSync(logFd, c as any); } catch {} };
	await new Promise<void>((resolve, reject) => {
		let child: ChildProcess;
		try { child = spawn(command, args, { cwd, detached: process.platform !== "win32", stdio: ["ignore", "pipe", "pipe"], env: process.env }); } catch (e) { closed = true; closeSync(logFd); reject(e); return; }
		activeChild = child;
		const out = (c: Buffer) => { writeChunk(c); progress?.onChunk(c); };
		child.stdout?.on("data", out); child.stderr?.on("data", out);
		const finish = (err?: Error) => { if (activeChild === child) activeChild = undefined; progress?.flush(); if (!closed) { closed = true; closeSync(logFd); } err ? reject(err) : resolve(); };
		child.on("error", finish); child.on("close", (code, signal) => code === 0 ? finish() : finish(new Error(`${label} failed (${signal ? `signal ${signal}` : `exit ${code}`}); see ${LOG_FILE}`)));
	});
}
function killActiveChild() { const child = activeChild; if (!child?.pid) return; try { process.kill(process.platform === "win32" ? child.pid : -child.pid, "SIGTERM"); } catch {} }

async function detectBackends(): Promise<Backend[]> {
	const backends = new Set<Backend>();
	if (platform() === "darwin" && arch() === "arm64") backends.add("metal");
	if (platform() === "win32" && await execCapture("nvidia-smi", [], 1500)) backends.add("cuda");
	if (await execCapture("rocminfo", [], 1500) || await execCapture("rocm-smi", [], 1500)) backends.add("rocm");
	if (await execCapture("vulkaninfo", ["--summary"], 1500)) backends.add("vulkan");
	backends.add("cpu");
	return [...backends];
}
async function chooseBackend(ctx?: { hasUI?: boolean; ui?: any }): Promise<Backend> {
	const forced = configString("LLAMA_CPP_BACKEND")?.toLowerCase() as Backend | undefined;
	if (forced) return forced;
	if (selectedBackend) return selectedBackend;
	const candidates = await detectBackends();
	const priority: Backend[] = ["metal", "cuda", "rocm", "vulkan", "cpu"];
	if (candidates.length > 1 && ctx?.hasUI && ctx.ui?.select) {
		const choice = await ctx.ui.select("Select llama.cpp runtime backend", candidates.map((b) => `${b}${b === "cpu" ? " (portable fallback)" : ""}`));
		if (choice) selectedBackend = choice.split(/\s+/, 1)[0] as Backend;
	}
	selectedBackend ??= priority.find((b) => candidates.includes(b)) ?? "cpu";
	return selectedBackend;
}
function assetScore(name: string, backend: Backend): number {
	const n = name.toLowerCase(); if (!n.endsWith(".zip") && !n.endsWith(".tar.gz")) return -1;
	if (platform() === "darwin" && !n.includes("macos")) return -1;
	if (platform() === "linux" && !(n.includes("linux") || n.includes("ubuntu"))) return -1;
	if (platform() === "win32" && !n.includes("win")) return -1;
	if (arch() === "arm64" && !(n.includes("arm64") || n.includes("aarch64"))) return -1;
	if (arch() === "x64" && !(n.includes("x64") || n.includes("x86_64"))) return -1;
	const has = (s: string) => n.includes(s);
	if (backend === "cuda" && !has("cuda")) return -1;
	if (backend === "rocm" && !has("rocm")) return -1;
	if (backend === "vulkan" && !has("vulkan")) return -1;
	if (backend === "metal" && !has("metal") && !has("macos")) return -1;
	if (backend === "cpu" && (has("cuda") || has("rocm") || has("vulkan") || has("openvino") || has("sycl") || has("hip") || has("opencl"))) return -1;
	return (has("avx2") ? 2 : 0) + (has("x64") ? 1 : 0);
}
async function latestReleaseAsset(backend: Backend): Promise<{ name: string; url: string }> {
	const forced = configString("LLAMA_CPP_RELEASE_URL");
	if (forced) return { name: basename(new URL(forced).pathname), url: forced };
	const res = await fetch("https://api.github.com/repos/ggml-org/llama.cpp/releases/latest", { headers: { "user-agent": "pi-llama-cpp" } });
	if (!res.ok) throw new Error(`GitHub release lookup failed: HTTP ${res.status}`);
	const release: any = await res.json();
	let best: any, bestScore = -1;
	for (const asset of release.assets ?? []) { const score = assetScore(String(asset.name), backend); if (score > bestScore) { best = asset; bestScore = score; } }
	if (!best) throw new Error(`No llama.cpp binary release asset found for backend ${backend}; set LLAMA_CPP_RELEASE_URL`);
	return { name: best.name, url: best.browser_download_url };
}
async function ensureRuntime(backend: Backend, onStatus?: StatusCallback): Promise<string> {
	const forced = configString("LLAMA_CPP_SERVER_BINARY"); if (forced) { await access(forced, constants.X_OK); return forced; }
	const dir = join(RUNTIME_DIR, backend); const binary = join(dir, process.platform === "win32" ? "llama-server.exe" : "llama-server");
	const manifest = await readJson<{ binary?: string }>(join(dir, "pi-llama-cpp-runtime.json"));
	if (manifest?.binary) { try { await access(manifest.binary, constants.X_OK); return manifest.binary; } catch {} }
	try { await access(binary, constants.X_OK); return binary; } catch {}
	await rm(dir, { recursive: true, force: true }); await mkdir(dir, { recursive: true });
	const asset = await latestReleaseAsset(backend); const zip = join(ROOT_DIR, asset.name);
	onStatus?.(`downloading llama.cpp ${backend} runtime`);
	await runLogged("curl", ["-L", "--fail", "--progress-bar", "-o", zip, asset.url], ROOT_DIR, `download ${asset.name}`, { onStatus, progressPrefix: "downloading llama.cpp" });
	onStatus?.("unpacking llama.cpp runtime");
	if (asset.name.toLowerCase().endsWith(".tar.gz")) await runLogged("tar", ["-xzf", zip, "-C", dir], ROOT_DIR, "unpack llama.cpp runtime", { onStatus, progressPrefix: "unpacking llama.cpp" });
	else await runLogged("unzip", ["-q", "-o", zip, "-d", dir], ROOT_DIR, "unpack llama.cpp runtime", { onStatus, progressPrefix: "unpacking llama.cpp" });
	const found = await findFile(dir, process.platform === "win32" ? "llama-server.exe" : "llama-server");
	if (!found) throw new Error(`Unpacked ${asset.name} but did not find llama-server`);
	await chmod(found, 0o755).catch(() => {}); await access(found, constants.X_OK);
	await writeJsonAtomic(join(dir, "pi-llama-cpp-runtime.json"), { backend, asset: asset.name, binary: found });
	return found;
}
async function findFile(dir: string, name: string): Promise<string | undefined> { for (const e of await readdir(dir, { withFileTypes: true }).catch(() => [])) { const p = join(dir, e.name); if (e.isFile() && e.name === name) return p; if (e.isDirectory()) { const r = await findFile(p, name); if (r) return r; } } }
async function chooseHfFile(spec: ModelSpec): Promise<string> {
	if (spec.file) return spec.file;
	const res = await fetch(`https://huggingface.co/api/models/${spec.repo}`, { headers: { "user-agent": "pi-llama-cpp" } });
	if (!res.ok) throw new Error(`Hugging Face model lookup failed for ${spec.repo}: HTTP ${res.status}. Set ${modelCatalogEntry(spec.id).prefix}_REPO/FILE.`);
	const data: any = await res.json();
	const files = (data.siblings ?? []).map((s: any) => String(s.rfilename ?? "")).filter((f: string) => f.toLowerCase().endsWith(".gguf"));
	const q = spec.quant.toLowerCase();
	return files.find((f: string) => f.toLowerCase().includes(q)) ?? files.find((f: string) => /q4_k_m/i.test(f)) ?? files[0] ?? (() => { throw new Error(`No .gguf files found in ${spec.repo}`); })();
}
async function ensureModel(spec: ModelSpec, onStatus?: StatusCallback): Promise<string> {
	await mkdir(MODELS_DIR, { recursive: true });
	const file = await chooseHfFile(spec); const out = join(MODELS_DIR, `${spec.id}-${basename(file)}`);
	try { await access(out, constants.R_OK); return await realpath(out); } catch {}
	const url = `https://huggingface.co/${spec.repo}/resolve/main/${file.split("/").map(encodeURIComponent).join("/")}?download=true`;
	onStatus?.(`downloading ${spec.id} ${spec.quant} model`);
	await runLogged("curl", ["-L", "--fail", "-C", "-", "--progress-bar", "-o", out, url], ROOT_DIR, `download ${spec.repo}/${file}`, { onStatus, progressPrefix: `downloading ${spec.id}` });
	await access(out, constants.R_OK); return await realpath(out);
}

async function isLockStale() { try { const info = await stat(LOCK_DIR); return Date.now() - info.mtimeMs > 60_000; } catch { return true; } }
async function withLock<T>(fn: () => Promise<T>, timeoutMs = LOCK_TIMEOUT_MS): Promise<T> { await mkdir(ROOT_DIR, { recursive: true }); const started = Date.now(); while (true) { try { await mkdir(LOCK_DIR); await writeJsonAtomic(join(LOCK_DIR, "owner.json"), { pid: process.pid, processStart: await getOwnProcessStart() }); break; } catch (e: any) { if (e?.code !== "EEXIST") throw e; if (await isLockStale()) { await rm(LOCK_DIR, { recursive: true, force: true }); continue; } if (Date.now() - started > timeoutMs) throw new Error(`Timed out waiting for llama.cpp lifecycle lock at ${LOCK_DIR}`); await sleep(150); } } try { return await fn(); } finally { await rm(LOCK_DIR, { recursive: true, force: true }); } }
async function ensureDirs() { await mkdir(CLIENT_DIR, { recursive: true }); }
async function touchLease() { const now = Date.now(); const lease: Lease = { managedBy: MANAGED_BY, usesLlamaCpp: true, pid: process.pid, processStart: await getOwnProcessStart(), cwd: process.cwd(), startedAt: leaseStartedAt, updatedAt: now, updatedAtIso: new Date(now).toISOString() }; await writeJsonAtomic(LEASE_FILE, lease); }
function startHeartbeat() { if (heartbeat) clearInterval(heartbeat); heartbeat = setInterval(() => void touchLease().catch(() => {}), HEARTBEAT_MS); heartbeat.unref?.(); }
function stopHeartbeat() { if (heartbeat) clearInterval(heartbeat); heartbeat = undefined; }
async function pruneLeases() { await mkdir(CLIENT_DIR, { recursive: true }); const entries = await readdir(CLIENT_DIR).catch(() => [] as string[]); for (const e of entries) { if (!e.endsWith(".json")) continue; const file = join(CLIENT_DIR, e); const lease = await readJson<Lease>(file); const info = await stat(file).catch(() => undefined); const live = lease?.pid && isPidAlive(lease.pid) && lease.processStart === await processStart(lease.pid); if (!live || !info || Date.now() - info.mtimeMs > LEASE_TTL_MS) await removeFile(file); } }
async function ensureWatchdog() { if (watchdogStarted) return; await access(WATCHDOG_SCRIPT, constants.F_OK); const logFd = openSync(LOG_FILE, "a"); try { const child = spawn("/bin/sh", [WATCHDOG_SCRIPT, ROOT_DIR], { detached: true, stdio: ["ignore", logFd, logFd], env: { ...process.env, LLAMA_CPP_DIR: ROOT_DIR, LLAMA_CPP_CLIENT_DIR: CLIENT_DIR, LLAMA_CPP_STATE_FILE: STATE_FILE, LLAMA_CPP_LOG_FILE: LOG_FILE, LLAMA_CPP_LEASE_TTL_S: String(Math.ceil(LEASE_TTL_MS / 1000)), LLAMA_CPP_WATCHDOG_POLL_S: "2", LLAMA_CPP_SHUTDOWN_GRACE_S: String(Math.ceil(SHUTDOWN_GRACE_MS / 1000)) } }); child.unref(); watchdogStarted = true; } finally { closeSync(logFd); } }
async function activateLease() { await ensureDirs(); await touchLease(); await pruneLeases(); await ensureWatchdog(); startHeartbeat(); }
async function readState() { return readJson<ServerState>(STATE_FILE); }
async function clearState() { await removeFile(STATE_FILE); }
async function checkHttpReady() { const c = new AbortController(); const t = setTimeout(() => c.abort(), HTTP_CHECK_TIMEOUT_MS); try { const r = await fetch(`${API_BASE_URL}/models`, { signal: c.signal }); return r.ok; } catch { return false; } finally { clearTimeout(t); } }
async function checkReadyForModel(modelId: ModelKey) { if (!(await checkHttpReady())) return false; return (await readState())?.modelId === modelId; }
async function waitForPidExit(pid: number, timeoutMs: number) { const end = Date.now() + timeoutMs; while (Date.now() < end) { if (!isPidAlive(pid)) return true; await sleep(500); } return !isPidAlive(pid); }
async function stopServerPidLocked(pid: number, reason: string) { await appendLog(`\n[${new Date().toISOString()}] ${reason}; stopping llama-server pid=${pid}\n`); try { process.kill(pid, "SIGTERM"); } catch {} if (!(await waitForPidExit(pid, SHUTDOWN_GRACE_MS))) { try { process.kill(pid, "SIGKILL"); } catch {} } await clearState(); }
function serverArgs(spec: ModelSpec, modelPath: string): string[] {
	const extra = (configString("LLAMA_CPP_SERVER_ARGS", "--parallel 1 --timeout 600") ?? "").split(/\s+/).filter(Boolean);
	return ["--host", "127.0.0.1", "--port", "8080", "--model", modelPath, "--ctx-size", String(spec.contextWindow), "--n-gpu-layers", configString("LLAMA_CPP_N_GPU_LAYERS", "999")!, "--jinja", "--reasoning", "off", ...extra];
}
async function startServerLocked(binary: string, backend: Backend, spec: ModelSpec, modelPath: string) { const args = serverArgs(spec, modelPath); await appendLog(`\n[${new Date().toISOString()}] start llama-server (${backend}, ${spec.id}, ${formatBytes(totalmem())} RAM)\n$ ${[binary, ...args].map(shellQuote).join(" ")}\n`); const logFd = openSync(LOG_FILE, "a"); let pid: number | undefined; try { const child = spawn(binary, args, { cwd: dirname(binary), detached: true, stdio: ["ignore", logFd, logFd], env: process.env }); child.unref(); pid = child.pid; } finally { closeSync(logFd); } if (!pid) throw new Error("failed to start llama-server"); const now = Date.now(); await writeJsonAtomic(STATE_FILE, { managedBy: MANAGED_BY, pid, baseUrl: API_BASE_URL, port: 8080, cwd: dirname(binary), binary, args, backend, modelId: spec.id, modelPath, startedAt: now, startedAtIso: new Date(now).toISOString() } satisfies ServerState); }
async function waitForServerReady(modelId: ModelKey, onStatus?: StatusCallback) { const started = Date.now(); let last = 0; while (Date.now() - started < READY_TIMEOUT_MS) { if (runtimeDisposed || shuttingDown) return; if (await checkReadyForModel(modelId)) return; const state = await readState(); if (state?.pid && !isPidAlive(state.pid)) throw new Error(`llama-server exited before becoming ready; see ${LOG_FILE}`); if (Date.now() - last > 10000) { onStatus?.(`llama-server starting (${Math.round((Date.now() - started) / 1000)}s)`); last = Date.now(); } await sleep(1000); } throw new Error(`Timed out waiting for llama-server at ${API_BASE_URL}; see ${LOG_FILE}`); }
async function ensureServerInner(modelId: ModelKey, ctx: any, onStatus?: StatusCallback) { const spec = modelSpec(modelId); let stoppingPid: number | undefined; await withLock(async () => { await activateLease(); const state = await readState(); if (state?.pid && isPidAlive(state.pid) && await looksLikeServer(state.pid)) { if (state.stopping) { stoppingPid = state.pid; return; } if (state.modelId === modelId) return; onStatus?.(`switching llama-server to ${modelId}`); await stopServerPidLocked(state.pid, `switch to ${modelId}`); } else if (state?.pid) await clearState(); const backend = await chooseBackend(ctx); const binary = await ensureRuntime(backend, onStatus); const modelPath = await ensureModel(spec, onStatus); onStatus?.(`starting llama-server (${backend}, ${modelId})`); await startServerLocked(binary, backend, spec, modelPath); }, STARTUP_LOCK_TIMEOUT_MS); if (stoppingPid) { await waitForPidExit(stoppingPid, SHUTDOWN_GRACE_MS); return ensureServerInner(modelId, ctx, onStatus); } await waitForServerReady(modelId, onStatus); }
function ensureServer(modelId: ModelKey, ctx: any, onStatus?: StatusCallback) { if (startupPromise) { if (startupModelId === modelId) return startupPromise; return startupPromise.catch(() => {}).then(() => ensureServer(modelId, ctx, onStatus)); } startupModelId = modelId; const p = ensureServerInner(modelId, ctx, onStatus).finally(() => { if (startupPromise === p) { startupPromise = undefined; startupModelId = undefined; } }); startupPromise = p; return p; }

function registerProvider(pi: ExtensionAPI) {
	pi.registerProvider(PROVIDER_ID, { name: "llama.cpp local", baseUrl: API_BASE_URL, api: PROVIDER_API, apiKey: configString("LLAMA_CPP_API_KEY", "llama-cpp-local"), compat: { supportsStore: false, supportsDeveloperRole: false, supportsReasoningEffort: false, supportsUsageInStreaming: true, maxTokensField: "max_tokens", supportsStrictMode: false }, models: allModels().map((m) => ({ id: m.id, name: m.name, reasoning: false, input: ["text"], contextWindow: m.contextWindow, maxTokens: m.maxTokens, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 } })) } as any);
}
function registerCommand(pi: ExtensionAPI) { pi.registerCommand("llama-cpp", { description: "Show llama.cpp runtime status and log path", handler: async (_args, ctx) => { const state = await readState(); ctx.ui.notify(state?.pid && isPidAlive(state.pid) ? `llama-server pid=${state.pid}, model=${state.modelId}, backend=${state.backend}; log: ${LOG_FILE}` : `llama-server is not running; log: ${LOG_FILE}`, "info"); } }); }

export default function (pi: ExtensionAPI) {
	runtimeDisposed = false; shuttingDown = false; leaseStartedAt = Date.now(); watchdogStarted = false; startupPromise = undefined; startupModelId = undefined; activeChild = undefined;
	registerProvider(pi); registerCommand(pi);
	pi.on("resources_discover", () => ({ skillPaths: [join(EXTENSION_DIR, "pi-llama-cpp-config", "SKILL.md")] }));
	pi.on("before_provider_request", async (_event, ctx) => {
		if (ctx.model?.provider !== PROVIDER_ID) return;
		const modelId = modelKeyForModelId(ctx.model?.id); if (!modelId) return;
		const ready = await checkReadyForModel(modelId); let last: string | undefined;
		const notify = ready ? undefined : (message?: string) => { if (!message || message === last || /^llama-server starting/.test(message)) return; last = message; ctx.ui.notify(message, "info"); };
		try { notify?.("preparing llama.cpp runtime"); await ensureServer(modelId, ctx, notify); if (!ready) ctx.ui.notify("llama-server ready", "info"); } catch (error) { ctx.ui.notify(`llama.cpp startup failed: ${describeError(error)}`, "error"); throw error; }
	});
	pi.on("session_shutdown", async (event, ctx) => { runtimeDisposed = true; stopHeartbeat(); killActiveChild(); if (startupPromise) await Promise.race([startupPromise.catch(() => {}), sleep(5000)]).catch(() => {}); if (event.reason !== "quit") return; shuttingDown = true; try { await removeFile(LEASE_FILE); } catch (error) { ctx.ui.notify(`llama.cpp shutdown failed: ${describeError(error)}`, "error"); } });
}
