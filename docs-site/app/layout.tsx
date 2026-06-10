import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import { RootProvider } from 'fumadocs-ui/provider/next';
import 'fumadocs-ui/style.css';
import './global.css';

export const metadata: Metadata = {
  metadataBase: new URL('https://docs.usejuno.co'),
  title: {
    default: 'Juno Docs',
    template: '%s | Juno Docs',
  },
  description:
    'Official docs for Juno, a local voice-writing app for Mac with live transcription, Voice Actions, Voice Commands, private offline dictation after setup, and no subscription tier.',
  alternates: {
    types: {
      'text/plain': '/llms.txt',
    },
  },
  robots: {
    index: true,
    follow: true,
    googleBot: {
      index: true,
      follow: true,
      'max-snippet': -1,
      'max-image-preview': 'large',
      'max-video-preview': -1,
    },
  },
  openGraph: {
    title: 'Juno Docs',
    description:
      'Official docs for Juno: setup, privacy, local models, Voice Actions, Voice Commands, and troubleshooting.',
    url: 'https://docs.usejuno.co/docs',
    siteName: 'Juno Docs',
    type: 'website',
  },
};

export default function Layout({ children }: { children: ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body>
        <RootProvider>{children}</RootProvider>
      </body>
    </html>
  );
}
