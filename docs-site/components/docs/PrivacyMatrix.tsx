export type PrivacyMatrixRow = {
  scenario: string;
  context: string;
  clipboardContext: string;
  history: string;
  recordings: string;
  learning: string;
  paste: string;
};

export function PrivacyMatrix({ rows }: { rows: PrivacyMatrixRow[] }) {
  return (
    <div className="juno-data-table-wrap">
      <table className="juno-data-table">
        <thead>
          <tr>
            {['Scenario', 'Context', 'Clipboard context', 'History', 'Recordings', 'Learning', 'Paste'].map((heading) => (
              <th key={heading}>{heading}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={row.scenario}>
              <td>{row.scenario}</td>
              <td>{row.context}</td>
              <td>{row.clipboardContext}</td>
              <td>{row.history}</td>
              <td>{row.recordings}</td>
              <td>{row.learning}</td>
              <td>{row.paste}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
