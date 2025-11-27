import type { LiveProvider, LiveEvent } from "@refinedev/core";
import { io } from "socket.io-client";

const WS_URL = process.env.BACKEND_WS_URL || "http://localhost:4000";

export const socket = io(WS_URL, {
  transports: ["websocket"],
});

export const liveProvider: LiveProvider = {
  subscribe: ({ channel, callback }) => {
    if (!channel.startsWith("resources/")) {
      return { unsubscribe: () => {} };
    }

    const handler = (event: any) => {
      const liveEvent: LiveEvent = {
        channel,
        type: event.type ?? "created",
        payload: event.payload ?? event,
        date: new Date(event.date ?? Date.now()),
      };
      callback(liveEvent);
    };

    socket.on(channel, handler);

    // Optional: log when actually connected
    if (socket.connected) {
      console.log("Live subscribed to:", channel);
    } else {
      socket.once("connect", () => {
        console.log("WebSocket connected — Live subscribed to:", channel);
      });
    }

    return {
      unsubscribe: () => {
        console.log("Unsubscribing...");
        socket.off(channel, handler);
        // Optional: disconnect if no listeners left
      },
    };
  },

  unsubscribe: ({ channel }) => {
    socket.removeAllListeners(channel);
  },

  publish: ({ channel, type, payload }: LiveEvent) => {
    socket.emit(channel, { type, payload });
  },
};
