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
  async parseJsonResponse(response) {
    const data = await response.json();
    const text = data?.output?.find(item => item.type === "message")
      ?.content?.find(item => item.type === "output_text")?.text ?? null;
    return { data, text };
  }
  async infer(payload) {
    return await this.send({ ...payload, stream: false });
  }

  async stream(payload) {
    return await this.send({ ...payload, stream: true });
  }
}