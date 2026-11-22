export function SurrealStudy({ type }: { type: number }) {
  if (type === 0) {
    return (
      <svg viewBox="0 0 420 470" fill="none" aria-hidden="true">
        <ellipse cx="220" cy="403" rx="132" ry="18" fill="#412b36" opacity=".09" />
        <circle cx="273" cy="94" r="52" fill="#c56948" />
        <path
          d="M225 77c15 6 55 7 94 0m-97 14c28 8 64 9 103 1m-100 16c28 8 61 8 94 0m-87 15c26 6 50 6 78-1"
          stroke="#7f442f"
          strokeWidth=".7"
          opacity=".5"
        />
        {Array.from({ length: 10 }, (_, index) => {
          const x = 66 + index * 14;
          const y = 369 - index * 23;
          return (
            <g key={index}>
              <path d={`M${x} ${y}h82v20h-82Z`} fill="#dbc8aa" stroke="#68534b" strokeWidth=".6" />
              <path
                d={`m${x} ${y} 62-28h82l-62 28Z`}
                fill="#f6edda"
                stroke="#68534b"
                strokeWidth=".6"
              />
              <path
                d={`m${x + 82} ${y} 62-28v20l-62 28Z`}
                fill="#b6998a"
                stroke="#68534b"
                strokeWidth=".6"
              />
            </g>
          );
        })}
        <path
          d="M78 102c-24 28-17 61 16 79-49 4-62-65-16-79Z"
          fill="#f8eed1"
          stroke="#9a8175"
          strokeWidth=".6"
        />
        <path d="m335 222 3 9 9 3-9 3-3 9-3-9-9-3 9-3Z" fill="#8a6458" />
      </svg>
    );
  }

  if (type === 1) {
    return (
      <svg viewBox="0 0 420 470" fill="none" aria-hidden="true">
        <ellipse cx="231" cy="398" rx="127" ry="18" fill="#412b36" opacity=".12" />
        <path d="M100 383V158a110 110 0 0 1 220 0v225Z" fill="#a48e9f" stroke="#675566" />
        <path
          d="M120 383V159a90 90 0 0 1 180 0v224Z"
          fill="#e9d7b9"
          stroke="#675566"
          strokeWidth=".7"
        />
        <path d="M145 383V188a65 65 0 0 1 130 0v195Z" fill="#614b63" />
        <path d="M165 383V219a45 45 0 0 1 90 0v164Z" fill="#d6b4ab" />
        <path d="M182 383V249a28 28 0 0 1 56 0v134Z" fill="#50435f" />
        <circle cx="210" cy="281" r="15" fill="#e9aa68" />
        <path
          d="M182 291h56m-56 7h56m-56 8h56m-56 10h56"
          stroke="#c49aa9"
          strokeWidth=".7"
          opacity=".6"
        />
        <path
          d="M102 112c-7-15-22-15-28-4-17-6-29 4-29 16h84c1-12-14-21-27-12Z"
          fill="#f4e8d5"
          stroke="#a68b87"
          strokeWidth=".6"
        />
        <path
          d="M302 242c-7-15-22-15-28-4-17-6-29 4-29 16h84c1-12-14-21-27-12Z"
          fill="#f4e8d5"
          stroke="#a68b87"
          strokeWidth=".6"
        />
        <path d="m117 384 150 58m32-58 67 28" stroke="#90788c" opacity=".3" />
      </svg>
    );
  }

  return (
    <svg viewBox="0 0 420 470" fill="none" aria-hidden="true">
      <ellipse cx="214" cy="400" rx="129" ry="17" fill="#412b36" opacity=".12" />
      <path d="m107 362 93 37 119-53-91-37Z" fill="#b69b87" stroke="#6c554c" strokeWidth=".7" />
      <path
        d="m107 362v22l93 37v-22Zm93 37 119-53v22l-119 53Z"
        fill="#d5baa1"
        stroke="#6c554c"
        strokeWidth=".7"
      />
      <path
        d="M147 354V199a67 67 0 0 1 134 0v155h-32V203a35 35 0 0 0-70 0v151Z"
        fill="#e9d5b4"
        stroke="#7d6657"
        strokeWidth=".8"
      />
      <path d="M147 354h32V203a35 35 0 0 1 22-32c-31 0-54 22-54 52Z" fill="#c7aa94" />
      <path d="m178 76 55-24 58 24-56 27Z" fill="#f0d4a6" stroke="#8e6650" strokeWidth=".8" />
      <path d="m178 76 57 27v61l-57-25Z" fill="#c77e58" stroke="#8e6650" strokeWidth=".8" />
      <path d="m235 103 56-27v64l-56 24Z" fill="#9e674e" stroke="#8e6650" strokeWidth=".8" />
      <path d="m192 124 28 12m-28-21 28 12m-28-21 28 12" stroke="#8e5546" opacity=".5" />
      <circle cx="99" cy="212" r="24" fill="#af6349" />
      <path d="M81 207c8 3 23 3 37 0m-41 9c13 4 26 4 45 0" stroke="#7d4b3b" opacity=".5" />
      <path d="m314 188 3 10 10 3-10 3-3 10-3-10-10-3 10-3Z" fill="#8a6458" />
    </svg>
  );
}
