export interface ProcessedTransaction {
  txFraud: boolean;
  txFraudScenario: number;
  transactionId: number;
  txDatetime: Date;
  customerId: number;
  terminalId: number;
  txAmount: number;
  txTimeSeconds: number;
  txTimeDays: number;
}
