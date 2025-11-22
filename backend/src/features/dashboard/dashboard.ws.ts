import { getIO } from "../../core/ws-server.js";
import { type ProcessedTransaction } from "../../core/fraud-analysis/fraud-analysis.types.js";

export function pushProcessedTransactionToWebSocketClient(
  processedTransaction: ProcessedTransaction,
  _ioClientId: string // TODO: implement when user auth is done
) {
  getIO().emit("newTransaction", processedTransaction);
}
