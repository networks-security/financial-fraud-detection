import { type Transaction } from "../../new-transaction/schemas/transactions.types.ts";
import { type ProcessedTransaction } from "../../features/dashboard/dashboard.types.ts";

export function transformTransactionIntoProcessedTransaction(
  transaction: Transaction,
  txFraud: boolean = false
): ProcessedTransaction {
  return {
    transactionId: transaction.transactionId,
    txDatetime: transaction.txDatetime,
    customerId: transaction.customerId,
    terminalId: transaction.terminalId,
    txAmount: transaction.txAmount,
    txTimeSeconds: transaction.txTimeSeconds,
    txTimeDays: transaction.txTimeDays,

    txFraud: txFraud,
    txFraudScenario: 0,
  };
}
