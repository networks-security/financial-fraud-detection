/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      // Basic redirect
      {
        source: "/dashboard",
        destination: "/dashboard/transactions",
        permanent: true,
      },
    ];
  },
};

export default nextConfig;
