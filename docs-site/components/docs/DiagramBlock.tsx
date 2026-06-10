import type { ReactNode } from 'react';

export function DiagramBlock(props: {
  title: string;
  caption: string;
  takeaway: string;
  children: ReactNode;
  sourceFiles?: string[];
}) {
  const { title, caption, takeaway, children } = props;

  return (
    <figure className="juno-diagram-block">
      <figcaption>
        <strong>{title}</strong>
        <div>{caption}</div>
      </figcaption>
      <div>{children}</div>
      <p><strong>Takeaway:</strong> {takeaway}</p>
    </figure>
  );
}
