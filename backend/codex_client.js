import {
  transformRequestForCodex,
  createCodexHeaders,
  handleSuccessResponse,
  handleErrorResponse,
} from "./lib/request/fetch-helpers.js";

export class CodexClient {
  constructor(authStore, config = {}) {
    this.authStore = authStore;
    this.config = config;
  }

  async getAuth() {
    return await this.authStore.load();
  }

  async login() {
    console.log("login() 尚未實作");
  }

  async refreshIfNeeded() {
    const auth = await this.authStore.load();
    return auth;
  }

  async send(payload) {
    const auth = await this.refreshIfNeeded();

    const init = {
      method: "POST",
      headers: {
        "content-type": "application/json"
      },
      body: JSON.stringify(payload)
    };

    const url = "https://chatgpt.com/backend-api/codex/responses";

    const transformed = await transformRequestForCodex(
      init,
      url,
      { global: {}, models: {} },
      this.config.codexMode ?? true
    );

    const finalInit = transformed?.updatedInit ?? init;

    const headers = createCodexHeaders(
      finalInit,
      auth?.accountId ?? "",
      auth?.access ?? "",
      {
        model: transformed?.body?.model
      }
    );

    const response = await fetch(url, {
      ...finalInit,
      headers
    });

    if (!response.ok) {
      return await handleErrorResponse(response);
    }

    return await handleSuccessResponse(response, payload.stream === true);
  }

  _extractTextFromJsonPayload(data) {
    return data?.output?.find(item => item.type === "message")
      ?.content?.find(item => item.type === "output_text")?.text ?? null;
  }

  _parseSseBody(body) {
    let text = "";

    for (const line of body.split(/\r?\n/)) {
      if (!line.startsWith("data: ")) {
        continue;
      }

      const raw = line.slice(6).trim();

      if (!raw || raw === "[DONE]") {
        continue;
      }

      try {
        const data = JSON.parse(raw);

        if (data?.type === "response.output_text.delta") {
          text += data.delta ?? "";
          continue;
        }

        const outputText = this._extractTextFromJsonPayload(data);
        if (outputText) {
          text = outputText;
        }
      } catch {
        continue;
      }
    }

    return { data: null, text: text || null };
  }

  async parseJsonResponse(response) {
    const contentType = response.headers.get("content-type") || "";
    if (contentType.toLowerCase().includes("text/html")) {
      const body = await response.text();
      const preview = body.slice(0, 300).replace(/\s+/g, " ").trim();
      throw new Error(
        `Expected JSON response but received HTML. content-type=${contentType} body_preview=${preview}`
      );
    }

    const body = await response.text();
    const trimmedBody = body.trimStart();

    if (trimmedBody.startsWith("<html") || trimmedBody.startsWith("<!doctype html")) {
      const preview = body.slice(0, 300).replace(/\s+/g, " ").trim();
      throw new Error(
        `Expected JSON response but received HTML body. content-type=${contentType || "unknown"} body_preview=${preview}`
      );
    }

    if (trimmedBody.startsWith("event:")) {
      return this._parseSseBody(trimmedBody);
    }

    const data = JSON.parse(body);
    const text = this._extractTextFromJsonPayload(data);
    return { data, text };
  }

  async infer(payload) {
    return await this.send({ ...payload, stream: false });
  }

  async stream(payload) {
    return await this.send({ ...payload, stream: true });
  }
}
