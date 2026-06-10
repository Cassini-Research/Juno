export type FlowStep = {
  title: string;
  description: string;
  status?: 'required' | 'optional' | 'advanced';
};

export function FlowSteps({ steps }: { steps: FlowStep[] }) {
  return (
    <ol className="juno-flow-steps">
      {steps.map((step, index) => (
        <li key={`${step.title}-${index}`} className="juno-flow-step">
          <div className="juno-flow-step__meta">
            <span>Step {index + 1}</span>
            {step.status ? <span>{step.status}</span> : null}
          </div>
          <strong>{step.title}</strong>
          <p>{step.description}</p>
        </li>
      ))}
    </ol>
  );
}
