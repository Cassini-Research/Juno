import { createMDX } from 'fumadocs-mdx/next';

const withMDX = createMDX();

/** @type {import('next').NextConfig} */
const nextConfig = {
  async redirects() {
    return [
      {
        source: '/docs/start/after-onboarding',
        destination: '/docs/start/start-using-juno',
        permanent: true,
      },
    ];
  },
};

export default withMDX(nextConfig);
