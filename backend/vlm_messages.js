import fs from "fs";

import { CodexClient } from "./codex_client.js";
import { AuthStore } from "./auth_store.js";
import { authData } from "./auth_data.js";

function readStdin() {
  return new Promise((resolve, reject) => {
    let data = "";

    process.stdin.setEncoding("utf8");

    process.stdin.on("data", chunk => {
      data += chunk;
    });

    process.stdin.on("end", () => {
      resolve(data);
    });

    process.stdin.on("error", err => {
      reject(err);
    });
  });
}

function normalizeMessagesToResponsesInput(messages) {
  return messages.map(message => {
    const role = message.role || "user";
    const content = message.content;

    if (typeof content === "string") {
      return {
        role,
        content: [
          {
            type: "input_text",
            text: content
          }
        ]
      };
    }

    if (Array.isArray(content)) {
      return {
        role,
        content: content.map(item => {
          if (item.type === "text") {
            return {
              type: "input_text",
              text: String(item.text ?? "")
            };
          }

          if (item.type === "image_url") {
            return {
              type: "input_image",
              image_url: item.image_url?.url,
              detail: item.image_url?.detail || "high"
            };
          }

          return item;
        })
      };
    }

    return {
      role,
      content: [
        {
          type: "input_text",
          text: String(content ?? "")
        }
      ]
    };
  });
}

async function main() {
  try {
    const raw = await readStdin();

    if (!raw.trim()) {
      throw new Error("No stdin input received.");
    }

    const payload = JSON.parse(raw);
    const messages = payload.messages;

    if (!Array.isArray(messages)) {
      throw new Error("payload.messages must be an array.");
    }

    const store = new AuthStore();
    await store.save(authData);

    const client = new CodexClient(store, { codexMode: true });

    const requestPayload = {
      model: payload.model || "gpt-5.2",
      instructions:
        messages.find(message => message.role === "system")?.content || "",
      input: normalizeMessagesToResponsesInput(
        messages.filter(message => message.role !== "system")
      ),
      max_output_tokens: payload.max_output_tokens || 300
    };

    const response = await client.infer(requestPayload);

    if (!(response instanceof Response)) {
      throw new Error("VLM backend did not return a Response object.");
    }

    const parsed = await client.parseJsonResponse(response);
    const text =
      parsed?.text ??
      parsed?.output_text ??
      parsed?.content ??
      parsed?.response?.text ??
      "";

    let result;
    try {
      result = JSON.parse(String(text).trim());
    } catch {
      result = {
        decision: "unsure",
        confidence: "low",
        reasoning: String(text || "Model returned non-JSON output."),
        matched_conditions: [],
        missing_conditions: [],
        suggested_action: "yaw"
      };
    }

    console.log(
      JSON.stringify({
        success: true,
        result
      })
    );
  } catch (err) {
    console.log(
      JSON.stringify({
        success: false,
        error: err?.message || String(err)
      })
    );
  }
}

main();