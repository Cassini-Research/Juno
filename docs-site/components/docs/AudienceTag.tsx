export type Audience = 'user' | 'developer' | 'contributor' | 'security' | 'maintainer';

const labels: Record<Audience, string> = {
  user: 'User',
  developer: 'Developer',
  contributor: 'Contributor',
  security: 'Security',
  maintainer: 'Maintainer',
};

export function AudienceTag({ audience }: { audience: Audience | Audience[] }) {
  const list = Array.isArray(audience) ? audience : [audience];

  return (
    <span style={{ display: 'inline-flex', gap: 6, flexWrap: 'wrap' }}>
      {list.map((item) => (
        <span
          key={item}
          style={{
            border: '1px solid var(--color-fd-border)',
            borderRadius: 999,
            padding: '2px 8px',
            fontSize: 12,
            color: 'var(--color-fd-muted-foreground)',
          }}
        >
          {labels[item]}
        </span>
      ))}
    </span>
  );
}
