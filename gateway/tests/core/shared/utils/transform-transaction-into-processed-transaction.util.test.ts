import { transformTransactionIntoProcessedTransaction } from "../../../../src/core/shared/utils/transform-transaction-into-processed-transaction.util.ts";
import { type Transaction } from "../../../../src/core/new-transaction/schemas/transactions.types.ts";
import { type ProcessedTransaction } from "../../../../src/core/analyzed-transaction/schemas/analyzed-transaction.types.ts";
import { describe, it, expect } from "@jest/globals";

describe("transformTransactionIntoProcessedTransaction", () => {
  it("should transform a transaction into a processed transaction with default txFraud value", () => {
    const transaction: Transaction = {
      transactionId: 12345,
      txDatetime: new Date("2025-12-27T10:00:00Z"),
      customerId: 23,
      terminalId: 3,
      txAmount: 100.5,
      txTimeSeconds: 3600,
      txTimeDays: 1,
    };

    const expected: ProcessedTransaction = {
      transactionId: 12345,
      txDatetime: new Date("2025-12-27T10:00:00Z"),
      customerId: 23,
      terminalId: 3,
      txAmount: 100.5,
      txTimeSeconds: 3600,
      txTimeDays: 1,
      txFraud: false,
      txFraudScenario: 0,
    };

    const result = transformTransactionIntoProcessedTransaction(transaction);
    expect(result).toEqual(expected);
  });

  it("should transform a transaction into a processed transaction with txFraud set to true", () => {
    const transaction: Transaction = {
      transactionId: 12345,
      txDatetime: new Date("2025-12-27T10:00:00Z"),
      customerId: 23,
      terminalId: 3,
      txAmount: 100.5,
      txTimeSeconds: 3600,
      txTimeDays: 1,
    };

    const expected: ProcessedTransaction = {
      transactionId: 12345,
      txDatetime: new Date("2025-12-27T10:00:00Z"),
      customerId: 23,
      terminalId: 3,
      txAmount: 100.5,
      txTimeSeconds: 3600,
      txTimeDays: 1,
      txFraud: true,
      txFraudScenario: 0,
    };

    const result = transformTransactionIntoProcessedTransaction(
      transaction,
      true
    );
    expect(result).toEqual(expected);
  });

  it("should transform a transaction into a processed transaction with txFraud set to false", () => {
    const transaction: Transaction = {
      transactionId: 12345,
      txDatetime: new Date("2025-12-27T10:00:00Z"),
      customerId: 23,
      terminalId: 3,
      txAmount: 100.5,
      txTimeSeconds: 3600,
      txTimeDays: 1,
    };

    const expected: ProcessedTransaction = {
      transactionId: 12345,
      txDatetime: new Date("2025-12-27T10:00:00Z"),
      customerId: 23,
      terminalId: 3,
      txAmount: 100.5,
      txTimeSeconds: 3600,
      txTimeDays: 1,
      txFraud: false,
      txFraudScenario: 0,
    };

    const result = transformTransactionIntoProcessedTransaction(
      transaction,
      false
    );
    expect(result).toEqual(expected);
  });
});
