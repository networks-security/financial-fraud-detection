import { transformTransactionIntoProcessedTransaction } from "../../shared/utils/transform-transaction-into-processed-transaction.util.ts";
import type { ProcessedTransaction } from "../../analyzed-transaction/schemas/analyzed-transaction.types.ts";
import type { Transaction } from "../schemas/transactions.types.ts";
import { execFileSync } from "node:child_process";
import path from "node:path";

export default class FraudAnalysisService {
  /**
   * Calls a ML Pipeline process that analyzes the transaction.
   * @param transaction Transaction to analyze.
   * @returns A processed transaction.
   * @returns Null if encountered an error.
   */
  execute(transaction: Transaction): ProcessedTransaction | null {
    const transactionJsonString = JSON.stringify(transaction);

    // Absolute path
    const pythonScript = process.env.ML_PIPELINE_MAIN_PY
      ? path.resolve(process.env.ML_PIPELINE_MAIN_PY)
      : "";

    if (pythonScript == "" || !pythonScript) {
      console.log(
        "Could not find the path to the main python ML script in .env ",
        pythonScript
      );
      return null;
    }

    try {
      const stdout = execFileSync(
        "python3",
        [pythonScript, transactionJsonString],
        {
          stdio: "pipe",
          encoding: "utf8",
        }
      );

      const jsonMlResult = JSON.parse(stdout);
      const transactionFraudScore = jsonMlResult["fraud_probability"];
      let boolTransactionFraudScore = true;
      if (transactionFraudScore < 0.5) boolTransactionFraudScore = false;
      else boolTransactionFraudScore = true;

      // Success
      return transformTransactionIntoProcessedTransaction(
        transaction,
        boolTransactionFraudScore
      );
    } catch (err) {
      console.log(err);
      return null;
    }
  }
}
