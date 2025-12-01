import {
  DataProvider,
  GetListParams,
  GetListResponse,
  BaseRecord,
} from "@refinedev/core";
import simpleRestDataProvider from "@refinedev/simple-rest";
import axios from "axios";
import { auth } from "../../../../config/firebase-config";
import { getAuthTokenFromCookies } from "../../../../shared/utils/get-auth-token-from-cookies.util";

const API_URL: string = process.env.BACKEND_API_URL || "http://localhost:4000";

const axiosInstance = axios.create();

axiosInstance.interceptors.request.use(
  async (config) => {
    let token = await auth.currentUser?.getIdToken();
    if (!token) {
      console.log(
        "Getting auth token from cookies, because current user is not found..."
      );
      const res = await getAuthTokenFromCookies();

      if (!res.ok) {
        console.log(
          "An error occurred when retrieving auth token from cookies"
        );
        token = "";
      }

      console.log(
        token ? "Auth token from cookies is " + token : "No token found"
      );
    }

    if (token) {
      config.headers.Authorization = "Bearer " + token;
    }

    return config;
  },
  (error) => Promise.reject(error)
);

const rest = simpleRestDataProvider(API_URL, axiosInstance);

export const dataProvider: DataProvider = {
  ...rest,

  getList: async <TData extends BaseRecord = BaseRecord>(
    params: GetListParams
  ): Promise<GetListResponse<TData>> => {
    console.log("Fetching transactions, params:", params);
    const result = await rest.getList<TData>(params);
    console.log("Successfully fetched transactions:", result.data);
    return result;
  },
};
