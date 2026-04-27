// auth_store.js
export class AuthStore {
  constructor() {
    this.auth = null;
  }

  async load() {
    return this.auth;
  }

  async save(auth) {
    this.auth = auth;
  }
}