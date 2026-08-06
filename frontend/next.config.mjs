/** @type {import('next').NextConfig} */
const nextConfig = {
  // Required by the Dockerfile: emits a self-contained server with only the
  // traced dependencies, instead of shipping the whole node_modules tree.
  output: "standalone",

  reactStrictMode: true,
  poweredByHeader: false,

  async headers() {
    return [
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "X-Frame-Options", value: "DENY" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
          // A clinical demo has no business being indexed or surfaced in search.
          { key: "X-Robots-Tag", value: "noindex, nofollow" },
        ],
      },
    ];
  },
};

export default nextConfig;
