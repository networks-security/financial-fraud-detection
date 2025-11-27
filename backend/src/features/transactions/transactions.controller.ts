import { type Request, type Response } from "express";
import { pushNotificationViaWebsocket } from "../dashboard/dashboard.ws.ts";
// import { type Transaction } from "./transactions.types.ts";

/**
 * This endpoint handler receives requests from the banking application.
 * @param req stores new transaction in res.body
 */
export async function processNewTransaction(_req: Request, res: Response) {
  // TODO: const newTransaction: Transaction = req.body;
  // TODO: call const txFraudResult = new MLPipelineService().execute(newTransaction);
  // TODO: transform Transaction into ProcessedTransaction (Transaction + txFraudResult)
  // TODO: save the processed transaction to the database and wait until it is saved before pushing the websocket notification

  pushNotificationViaWebsocket("IO_ID_NULL"); // NOTE: temp id until auth is implemented
  res.send("Transaction processed");
}
