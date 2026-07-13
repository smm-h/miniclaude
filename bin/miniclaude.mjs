#!/usr/bin/env node
import { spawn, execFileSync } from "child_process";

// Find the Python miniclaude binary
let binary;
try {
  binary = execFileSync("which", ["miniclaude"], { encoding: "utf-8" }).trim();
} catch {
  // Try common paths
  for (const path of ["/usr/local/bin/miniclaude", `${process.env.HOME}/.local/bin/miniclaude`]) {
    try {
      execFileSync("test", ["-x", path]);
      binary = path;
      break;
    } catch { /* continue */ }
  }
}

if (!binary) {
  console.error("Error: miniclaude Python package not found.");
  console.error("Install with: pip install miniclaude");
  process.exit(1);
}

const child = spawn(binary, process.argv.slice(2), { stdio: "inherit" });
child.on("close", (code) => process.exit(code ?? 1));
