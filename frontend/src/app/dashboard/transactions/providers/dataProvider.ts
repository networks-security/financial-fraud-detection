import {
  DataProvider,
  GetListParams,
  GetListResponse,
  BaseRecord,
} from "@refinedev/core";
import simpleRestDataProvider from "@refinedev/simple-rest";

const API_URL: string = process.env.BACKEND_API_URL || "http://localhost:4000";

const rest = simpleRestDataProvider(API_URL);

export const dataProvider: DataProvider = {
  ...rest,

  getList: async <TData extends BaseRecord = BaseRecord>(
    params: GetListParams
  ): Promise<GetListResponse<TData>> => {
    console.log("Fetching transaction:", params);
    const result = await rest.getList<TData>(params);
    console.log("Fetched transactions:", result.data);
    return result;
  },
};
