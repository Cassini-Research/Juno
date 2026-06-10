export function ScreenshotFrame(props: {
  src: string;
  alt: string;
  caption: string;
  surface?: string;
  version?: string;
  dateCaptured?: string;
  className?: string;
}) {
  const { src, alt, caption, surface, version, dateCaptured, className } = props;

  return (
    <figure className={['juno-screenshot-frame', className].filter(Boolean).join(' ')}>
      {/* eslint-disable-next-line @next/next/no-img-element */}
      <img src={src} alt={alt} />
      <figcaption>{caption}</figcaption>
    </figure>
  );
}
