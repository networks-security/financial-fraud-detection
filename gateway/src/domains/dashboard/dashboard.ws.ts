import { getIO } from "../../infra/websockets/ws-server.js";

export function pushNotificationViaWebsocket(
  ioClientId: string // TODO: implement when user auth is done
) {
  if (!ioClientId) {
    // TODO: auth check
    console.log("Unauthorized user");
    return;
  }
  console.log("Pushing a notification to the ws client with id, ", ioClientId);
  getIO().emit("resources/transactions", {
    type: "created",
  });
}
