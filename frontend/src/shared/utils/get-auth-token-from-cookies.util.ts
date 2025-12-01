export const getAuthTokenFromCookies = async () => {
  const res = await fetch("/api/get-auth-token");
  const token = await res.json();
  return token;
};
