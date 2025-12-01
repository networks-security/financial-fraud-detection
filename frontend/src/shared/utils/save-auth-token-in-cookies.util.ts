export const saveAuthTokenInCookies = async (token: string) => {
  return await fetch("/api/save-auth-token", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ token }),
  });
};
