import { describe, expect, it } from "vitest";
import { ApiError, wsUrlForExperiment } from "@/lib/api-client";

describe("wsUrlForExperiment", () => {
  it("builds a websocket URL with the experiment id", () => {
    expect(wsUrlForExperiment(42)).toBe("ws://localhost:8000/ws/experiments/42");
  });

  it("produces distinct URLs for distinct experiment ids", () => {
    expect(wsUrlForExperiment(1)).not.toBe(wsUrlForExperiment(2));
  });
});

describe("ApiError", () => {
  it("carries status and body through", () => {
    const error = new ApiError(404, { detail: "not found" });
    expect(error.status).toBe(404);
    expect(error.body).toEqual({ detail: "not found" });
    expect(error).toBeInstanceOf(Error);
  });
});
