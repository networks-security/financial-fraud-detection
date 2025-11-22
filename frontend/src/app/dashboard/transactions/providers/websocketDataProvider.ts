import type { LiveProvider, LiveEvent } from "@refinedev/core";
import { io } from "socket.io-client";

const WS_URL = process.env.BACKEND_WS_URL || "http://localhost:4000";

export const socket = io(WS_URL, {
  transports: ["websocket"],
});

export const liveProvider: LiveProvider = {
  subscribe: ({ channel, callback }): { unsubscribe: () => void } => {
    const handler = (event: any) => {
      const liveEvent: LiveEvent = {
        channel,
        type: event.type ?? "updated", // refine requires a type
        payload: event.payload,
        date: new Date(),
      };

      callback(liveEvent);
    };

    socket.on(channel, handler);

    return {
      unsubscribe: () => socket.off(channel, handler),
    };
  },

  unsubscribe: ({ channel }) => {
    socket.removeAllListeners(channel);
  },

  publish: ({ channel, type, payload }: LiveEvent) => {
    socket.emit(channel, { type, payload });
  },
};
