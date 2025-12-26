import { type Request, type Response } from "express";
import { pushNotificationViaWebsocket } from "../../../core/shared/utils/push-notification-via-websocket.ts";
import { type Transaction } from "../../../core/new-transaction/schemas/transactions.types.ts";
import FraudAnalysisService from "../../../core/new-transaction/services/fraud-analysis.service.ts";

/**
 * This endpoint handler receives requests from the banking application.
 * @param req stores new transaction in res.body
 */
export async function processNewTransaction(req: Request, res: Response) {
  console.log("New transaction processing request received");
  const newTransaction: Transaction = req.body;

  // TODO: validate data format of newTransaction

  const processedTransaction = new FraudAnalysisService().execute(
    newTransaction
  );
  console.log("Transaction after processing:", processedTransaction);

  if (!processedTransaction) {
    console.log("Failed to process the transaction");
    res.send("Failed to Process the Transaction: Server Side Error");
    return;
  }

  // TODO: save the processed transaction to the database and wait until it is saved before pushing the websocket notification

  pushNotificationViaWebsocket("IO_ID_NULL"); // NOTE: temp id until auth is implemented
  res.send("Transaction processed");
}
