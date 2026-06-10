const startCards = [
  {
    title: 'New install',
    body: 'Install Juno, grant permissions, and run your first dictation.',
    href: '/docs/start/first-launch',
    label: 'Onboarding',
  },
  {
    title: 'Start using Juno',
    body: 'Learn the daily writing loop and the main product surfaces.',
    href: '/docs/start/start-using-juno',
    label: 'Use',
  },
  {
    title: 'Quickstart',
    body: 'Dictate into a Mac text field in two minutes.',
    href: '/docs/start/quickstart',
    label: 'Fast path',
  },
];

const useCaseCards = [
  {
    title: 'App behavior',
    body: 'Set writing style, context, and privacy by app.',
    href: '/docs/use-juno/app-integrations',
    image: '/images/screenshots/live/real/per-app-writing-configured-real.png',
    alt: 'Juno Per-app writing screen with configured app behavior.',
  },
  {
    title: 'Transform text',
    body: 'Rewrite selected text, change tone, create bullets, or translate.',
    href: '/docs/use-juno/transformations',
    image: '/images/screenshots/live/real/styles-real.png',
    alt: 'Juno Styles screen showing built-in writing styles.',
  },
  {
    title: 'Create actions',
    body: 'Create reminders, notes, and Calendar alerts from speech.',
    href: '/docs/use-juno/actions',
    image: '/images/screenshots/live/real/actions-real.png',
    alt: 'Juno Voice Actions screen showing Reminders, Notes, and Alarms enabled.',
  },
];

const featureRows = [
  ['Dictate and insert', 'Start, stop, and insert final text.', '/docs/use-juno/speak-and-insert'],
  ['Voice Commands', 'Edit recent or selected text by voice.', '/docs/use-juno/voice-commands'],
  ['Dictionary & Memory', 'Add names, terms, snippets, and corrections.', '/docs/use-juno/dictionary-and-memory'],
  ['History', 'Search, copy, replay, reprocess, and delete sessions.', '/docs/use-juno/history'],
  ['Privacy', 'Control context, retention, and secure fields.', '/docs/privacy-and-data/overview'],
  ['Troubleshooting', 'Fix setup, audio, engine, and insertion issues.', '/docs/troubleshooting/overview'],
];

const comparisonRows = [
  ['System dictation', 'Basic speech-to-text.'],
  ['Cloud voice assistants', 'Command routing and hosted processing.'],
  ['Juno', 'Local voice writing inside your Mac apps.'],
];

export function DocsHome() {
  return (
    <main className="juno-home">
      <section className="juno-hero" aria-labelledby="juno-home-title">
        <div className="juno-hero__copy">
          <h1 id="juno-home-title">Voice writing for Mac apps.</h1>
          <p className="juno-hero__lead">
            Use one shortcut to dictate, revise, and insert polished text wherever you write.
          </p>
          <p className="juno-hero__meta">macOS 15+ · Apple Silicon · No account required</p>
          <div className="juno-hero__actions" aria-label="Primary documentation actions">
            <a className="juno-button juno-button--primary" href="/docs/start/start-using-juno">
              Start using Juno
            </a>
            <a className="juno-button" href="/docs/start/first-launch">
              Install Juno
            </a>
          </div>
        </div>

        <figure className="juno-hero-shot">
          <img
            src="/images/screenshots/home-real.png"
            alt="Juno Home screen showing readiness, Try It, daily stats, and sidebar navigation."
          />
        </figure>
      </section>

      <section className="juno-section juno-section--tight" aria-labelledby="paths-title">
        <div className="juno-section__head">
          <h2 id="paths-title">Start here</h2>
        </div>
        <div className="juno-path-grid">
          {startCards.map((item, index) => (
            <a className="juno-path-card" href={item.href} key={item.title}>
              <span className="juno-path-card__step">{String(index + 1).padStart(2, '0')}</span>
              <span className="juno-path-card__label">{item.label}</span>
              <strong>{item.title}</strong>
              <p>{item.body}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="juno-section" aria-labelledby="use-cases-title">
        <div className="juno-section__head">
          <h2 id="use-cases-title">Core workflows</h2>
        </div>
        <div className="juno-usecase-grid">
          {useCaseCards.map((card) => (
            <a className="juno-usecase-card" href={card.href} key={card.title}>
              <img src={card.image} alt={card.alt} />
              <span>{card.title}</span>
              <p>{card.body}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="juno-section" aria-labelledby="features-title">
        <div className="juno-section__head">
          <h2 id="features-title">Feature guides</h2>
        </div>
        <div className="juno-feature-list">
          {featureRows.map(([title, body, href]) => (
            <a href={href} className="juno-feature-row" key={title}>
              <strong>{title}</strong>
              <p>{body}</p>
            </a>
          ))}
        </div>
      </section>

      <section className="juno-section" aria-labelledby="positioning-title">
        <div className="juno-section__head">
          <h2 id="positioning-title">Local by design</h2>
        </div>
        <div className="juno-compare">
          {comparisonRows.map(([title, body]) => (
            <div className="juno-compare__row" key={title}>
              <strong>{title}</strong>
              <p>{body}</p>
            </div>
          ))}
        </div>
      </section>

      <section className="juno-home__footer" aria-labelledby="deep-title">
        <div>
          <h2 id="deep-title">Reference</h2>
        </div>
        <div className="juno-footer-links">
          <a href="/docs/troubleshooting/overview">Troubleshooting</a>
          <a href="/docs/architecture/overview">Architecture</a>
          <a href="/docs/reference/overview">Reference index</a>
          <a href="/docs/developers/repo-overview">Developer setup</a>
          <a href="/docs/privacy-and-data/overview">Privacy &amp; data</a>
        </div>
      </section>
    </main>
  );
}
