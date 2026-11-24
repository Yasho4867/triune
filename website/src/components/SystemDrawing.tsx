export function SystemDrawing({ type }: { type: number }) {
  if (type === 0) {
    return (
      <svg
        viewBox="0 0 680 420"
        fill="none"
        role="img"
        aria-label="Schematic of a visual model workspace connecting an input through attention, experts, and an output"
      >
        <defs>
          <pattern id="studio-dots" width="18" height="18" patternUnits="userSpaceOnUse">
            <circle cx="1" cy="1" r=".7" fill="currentColor" opacity=".18" />
          </pattern>
        </defs>
        <rect
          x="32"
          y="30"
          width="616"
          height="360"
          fill="var(--paper)"
          stroke="currentColor"
          strokeWidth="1"
        />
        <rect x="33" y="74" width="614" height="315" fill="url(#studio-dots)" />
        <path d="M32 74h616M146 74v316" stroke="currentColor" opacity=".3" />
        <circle cx="53" cy="52" r="3" fill="currentColor" />
        <circle cx="65" cy="52" r="3" stroke="currentColor" />
        <circle cx="77" cy="52" r="3" stroke="currentColor" />
        <text x="102" y="56" className="drawing-mono" fontSize="10" fill="currentColor">
          polysom studio / workspace
        </text>
        <text x="534" y="56" className="drawing-mono" fontSize="8" fill="currentColor">
          LOCAL FIRST
        </text>
        {["WORKSPACE", "Architecture", "Training", "Neuron scope", "Inference"].map((text, i) => (
          <text
            key={text}
            x="48"
            y={108 + i * 31}
            className="drawing-mono"
            fontSize={i === 0 ? "8" : "9"}
            fill="currentColor"
            opacity={i < 2 ? 1 : 0.4}
          >
            {text}
          </text>
        ))}
        <path
          d="M224 228h34v-77h48m-48 77v76h48m112-153h34v77h44m-78 76h34v-76"
          stroke="currentColor"
          strokeWidth="1.4"
        />
        <circle cx="258" cy="228" r="4" fill="currentColor" />
        <circle cx="452" cy="228" r="4" fill="currentColor" />
        <rect x="172" y="208" width="66" height="40" fill="var(--paper)" stroke="currentColor" />
        <text
          x="205"
          y="232"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="9"
          fill="currentColor"
        >
          INPUT
        </text>
        <rect x="306" y="122" width="112" height="58" fill="var(--blue)" />
        <text
          x="362"
          y="146"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="10"
          fill="white"
        >
          ATTENTION
        </text>
        <text
          x="362"
          y="164"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="8"
          fill="white"
          opacity=".6"
        >
          GATED LINEAR
        </text>
        <rect x="306" y="275" width="112" height="58" fill="var(--paper)" stroke="currentColor" />
        <text
          x="362"
          y="300"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="10"
          fill="currentColor"
        >
          EXPERTS
        </text>
        <text
          x="362"
          y="318"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="8"
          fill="currentColor"
          opacity=".6"
        >
          SPARSE MoE
        </text>
        <rect x="496" y="208" width="112" height="40" fill="var(--paper)" stroke="currentColor" />
        <text
          x="552"
          y="232"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="9"
          fill="currentColor"
        >
          OUTPUT
        </text>
        <path d="m286 147 5 4-5 4m0 145 5 4-5 4m195-80 5 4-5 4" stroke="currentColor" />
      </svg>
    );
  }

  if (type === 1) {
    return (
      <svg
        viewBox="0 0 680 420"
        fill="none"
        role="img"
        aria-label="Diagram of layer streaming from host memory through a smaller GPU working set"
      >
        <text x="72" y="62" className="drawing-mono" fontSize="10" fill="currentColor">
          HOST MEMORY
        </text>
        <text x="454" y="62" className="drawing-mono" fontSize="10" fill="currentColor">
          GPU WORKING SET
        </text>
        {Array.from({ length: 7 }, (_, i) => (
          <g key={i}>
            <path
              d={`M72 ${104 + i * 31} 260 ${104 + i * 31} 290 ${88 + i * 31} 102 ${88 + i * 31}Z`}
              fill={i === 3 ? "var(--blue)" : "var(--paper)"}
              stroke="currentColor"
            />
            <path
              d={`M72 ${104 + i * 31}v15h188v-15m0 15 30-16V${88 + i * 31}`}
              stroke="currentColor"
            />
          </g>
        ))}
        <path d="M322 196h95m-10-9 10 9-10 9" stroke="currentColor" strokeWidth="1.4" />
        <text
          x="363"
          y="178"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="8"
          fill="currentColor"
        >
          STREAM
        </text>
        <rect x="452" y="112" width="148" height="169" stroke="currentColor" />
        <rect x="465" y="125" width="122" height="64" fill="var(--blue)" />
        <rect
          x="465"
          y="201"
          width="122"
          height="64"
          stroke="currentColor"
          strokeDasharray="3 3"
          opacity=".5"
        />
        <text
          x="526"
          y="152"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="10"
          fill="white"
        >
          ACTIVE LAYER
        </text>
        <text
          x="526"
          y="171"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="8"
          fill="white"
          opacity=".6"
        >
          COMPUTE
        </text>
        <text
          x="526"
          y="237"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="9"
          fill="currentColor"
          opacity=".6"
        >
          PREFETCH
        </text>
        <path
          d="M526 303v26H161v-15m9 7-9-7-9 7"
          stroke="currentColor"
          strokeWidth="1"
          opacity=".5"
        />
        <text
          x="350"
          y="350"
          textAnchor="middle"
          className="drawing-mono"
          fontSize="9"
          fill="currentColor"
        >
          MORE MODEL. LESS RESIDENT MEMORY.
        </text>
      </svg>
    );
  }

  return (
    <svg
      viewBox="0 0 680 420"
      fill="none"
      role="img"
      aria-label="Conceptual diagram of expert optimization projected into a compact subspace"
    >
      <path d="M64 334h549M112 360V72" stroke="currentColor" opacity=".25" />
      <path
        d="m112 280 420 42 73-167-345-55Z"
        fill="var(--paper)"
        stroke="currentColor"
        opacity=".8"
      />
      {Array.from({ length: 9 }, (_, i) => (
        <path
          key={i}
          d={`M${130 + i * 46} ${282 + i * 4.6} ${270 + i * 35} ${102 + i * 5.6}`}
          stroke="currentColor"
          opacity=".1"
        />
      ))}
      {Array.from({ length: 52 }, (_, i) => {
        const angle = i * 2.399;
        const radius = Math.sqrt(i) * 14;
        return (
          <circle
            key={i}
            cx={330 + Math.cos(angle) * radius * 1.7}
            cy={213 + Math.sin(angle) * radius * 0.68}
            r={i % 8 === 0 ? "4" : "2.5"}
            fill="currentColor"
            opacity={i % 5 === 0 ? ".7" : ".28"}
          />
        );
      })}
      <circle cx="330" cy="213" r="57" stroke="currentColor" strokeDasharray="3 4" />
      <circle cx="330" cy="213" r="8" fill="var(--blue)" />
      <path d="m218 275 112-62 160 17m-15-6 15 6-15 6" stroke="currentColor" strokeWidth="2" />
      <path d="M330 213V93h114" stroke="currentColor" opacity=".5" />
      <text x="454" y="96" className="drawing-mono" fontSize="9" fill="currentColor">
        CENTROID
      </text>
      <text x="145" y="380" className="drawing-mono" fontSize="10" fill="currentColor">
        EXPERT SPACE
      </text>
      <text x="472" y="380" className="drawing-mono" fontSize="10" fill="currentColor">
        COMPACT STATE
      </text>
    </svg>
  );
}
