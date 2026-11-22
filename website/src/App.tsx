import { useEffect, useRef, useState } from "react";
import { SystemDrawing } from "./components/SystemDrawing";
import { SurrealStudy } from "./components/SurrealStudy";

const repository = "https://github.com/Yasho4867/triune";

const systems = [
  {
    name: "Polysom Studio",
    category: "The workspace",
    description:
      "From an architecture on a canvas to a model running on your machine. A native environment for training, fine-tuning, inspecting, and serving language models.",
    features: ["Visual model builder", "Live activation inspection", "Local-first workflows"],
    command: "polysom studio",
    link: `${repository}#launching-polysom-studio`,
    action: "Explore the studio",
  },
  {
    name: "Triune Engine",
    category: "The infrastructure",
    description:
      "The systems beneath the model. A hardware-aware training and inference stack with sparse experts, fused operators, and memory-conscious layer streaming.",
    features: ["Linear attention", "Sparse mixture of experts", "Layer streaming"],
    command: "import lucid_machines",
    link: `${repository}/blob/main/docs/FRAMEWORK.md`,
    action: "Inside the engine",
  },
  {
    name: "CentroidSteer",
    category: "The optimizer",
    description:
      "Rethinking how optimization uses memory. A partitioned optimizer combining centroid steering, subspace updates, and hardware-aware execution.",
    features: ["Low-memory state", "Subspace optimization", "PyTorch integration"],
    command: "lucid_machines.CentroidSteer",
    link: `${repository}/tree/main/papers/paper1_centroidsteer`,
    action: "Read the research",
  },
];

const programs = [
  {
    number: "01",
    title: "Intelligence that adapts.",
    label: "Adaptive computation",
    description:
      "Not every problem needs the same amount of computation. We study multi-exit architectures, calibrated early exits, and self-speculative decoding to make model depth a decision—not a constant.",
    detail: "Reflex → Limbic → Cortex",
    link: `${repository}/blob/main/benchmarks/DYNAMIC_EXITS_REPORT.md`,
    linkText: "Dynamic exits report",
  },
  {
    number: "02",
    title: "Context without the cost.",
    label: "Efficient sequence modeling",
    description:
      "Longer context should not require an ever-growing memory footprint. Our work explores gated linear attention and recurrent state updates, with attention to recall quality as well as throughput.",
    detail: "Linear attention / Recurrent state",
    link: `${repository}/blob/main/benchmarks/LONG_CONTEXT_REPORT.md`,
    linkText: "Long-context report",
  },
  {
    number: "03",
    title: "Capability within reach.",
    label: "Hardware-aware learning",
    description:
      "We work across optimizer state, weight precision, and layer streaming to make larger models practical on constrained hardware. The hardware budget is part of the research problem.",
    detail: "Quantization / Streaming / Optimization",
    link: `${repository}/blob/main/benchmarks/LAYER_STREAMING_REPORT.md`,
    linkText: "Layer-streaming report",
  },
];

function Arrow({ diagonal = true }: { diagonal?: boolean }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      aria-hidden="true"
      className={`arrow-icon${diagonal ? "" : " arrow-straight"}`}
    >
      <path d="M5 19 19 5M5 5h14v14" stroke="currentColor" strokeWidth="1.5" />
    </svg>
  );
}

function Symbol({ className = "" }: { className?: string }) {
  return (
    <svg viewBox="0 0 48 48" fill="none" className={className} aria-hidden="true">
      <path d="M3 25C13 10 35 10 45 25 35 40 13 40 3 25Z" stroke="currentColor" strokeWidth="1.5" />
      <path
        d="m8 17-3-4m11-1-2-5m10 3V4m8 8 2-5m6 10 3-4M14 37l-2 4m12-3v6m10-7 2 4"
        stroke="currentColor"
        strokeWidth="1.2"
      />
      <ellipse cx="24" cy="25" rx="8.5" ry="10.5" fill="currentColor" />
      <ellipse cx="24" cy="25" rx="3" ry="5.5" fill="var(--symbol-center, var(--paper))" />
      <circle cx="27" cy="21" r="1.8" fill="var(--symbol-center, var(--paper))" />
    </svg>
  );
}

function Brand({ footer = false, onNavigate }: { footer?: boolean; onNavigate?: () => void }) {
  return (
    <a
      className={`brand${footer ? " brand-footer" : ""}`}
      href="#top"
      aria-label="Lucid Machines home"
      onClick={onNavigate}
    >
      <Symbol className="brand-symbol" />
      <span className="brand-letters">
        <strong>lucid</strong>
        <span>machines</span>
      </span>
    </a>
  );
}

function SectionLabel({ number, children }: { number: string; children: React.ReactNode }) {
  return (
    <p className="section-label">
      <span className="label-number">{number}</span>
      {children}
    </p>
  );
}

export default function App() {
  const [menuOpen, setMenuOpen] = useState(false);
  const [activeSection, setActiveSection] = useState("");
  const [activeSystem, setActiveSystem] = useState(0);
  const [activeProgram, setActiveProgram] = useState(0);
  const [refracted, setRefracted] = useState(false);
  const [headerScrolled, setHeaderScrolled] = useState(false);
  const dialog = useRef<HTMLDialogElement>(null);
  const modalTrigger = useRef<HTMLButtonElement>(null);
  const mobileTrigger = useRef<HTMLButtonElement>(null);
  const systemTabs = useRef<(HTMLButtonElement | null)[]>([]);
  const system = systems[activeSystem];

  useEffect(() => {
    const updateHeader = () => setHeaderScrolled(window.scrollY > 80);
    updateHeader();
    window.addEventListener("scroll", updateHeader, { passive: true });
    return () => window.removeEventListener("scroll", updateHeader);
  }, []);

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) setActiveSection(entry.target.id);
        }
      },
      { rootMargin: "-15% 0px -65% 0px" },
    );
    document.querySelectorAll("main section[id]").forEach((section) => observer.observe(section));
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    if (!menuOpen) return;
    const dismiss = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setMenuOpen(false);
        mobileTrigger.current?.focus();
      }
    };
    const resize = () => {
      if (window.innerWidth >= 900) setMenuOpen(false);
    };
    document.addEventListener("keydown", dismiss);
    window.addEventListener("resize", resize);
    return () => {
      document.removeEventListener("keydown", dismiss);
      window.removeEventListener("resize", resize);
    };
  }, [menuOpen]);

  useEffect(() => {
    const element = dialog.current;
    if (!element) return;
    const unlock = () => {
      document.body.style.overflow = "";
    };
    element.addEventListener("close", unlock);
    return () => {
      unlock();
      element.removeEventListener("close", unlock);
    };
  }, []);

  const openBrief = () => {
    dialog.current?.showModal();
    document.body.style.overflow = "hidden";
  };

  const navItems = [
    { href: "junkie", text: "The model" },
    { href: "systems", text: "Our systems" },
    { href: "research", text: "Research" },
    { href: "company", text: "The company" },
  ];

  return (
    <>
      <a className="skip-link" href="#main">
        Skip to content
      </a>

      <header className={`site-header${headerScrolled ? " is-scrolled" : ""}`}>
        <div className="header-inner">
          <Brand />
          <nav className="desktop-nav" aria-label="Main navigation">
            {navItems.map((item) => (
              <a
                key={item.href}
                href={`#${item.href}`}
                className={activeSection === item.href ? "is-active" : ""}
                aria-current={activeSection === item.href ? "location" : undefined}
              >
                {item.text}
              </a>
            ))}
          </nav>
          <a className="header-source" href={repository} target="_blank" rel="noreferrer">
            Open source <Arrow />
          </a>
          <button
            ref={mobileTrigger}
            className={`mobile-toggle${menuOpen ? " is-open" : ""}`}
            type="button"
            aria-label={menuOpen ? "Close navigation" : "Open navigation"}
            aria-expanded={menuOpen}
            aria-controls="mobile-navigation"
            onClick={() => setMenuOpen(!menuOpen)}
          >
            <span />
            <span />
          </button>
        </div>
        {menuOpen && (
          <nav id="mobile-navigation" className="mobile-nav" aria-label="Mobile navigation">
            {navItems.map((item, index) => (
              <a key={item.href} href={`#${item.href}`} onClick={() => setMenuOpen(false)}>
                <span>0{index + 1}</span>
                {item.text}
                <Arrow />
              </a>
            ))}
            <a
              href={repository}
              target="_blank"
              rel="noreferrer"
              onClick={() => setMenuOpen(false)}
            >
              <span>↗</span>
              Open source
              <Arrow />
            </a>
          </nav>
        )}
      </header>

      <main id="main">
        <section id="top" className="hero-section">
          <div className="hero-overline page-gutter">
            <p>
              <span className="status-dot" /> Independent artificial intelligence research
            </p>
            <span className="hero-edition">Independent minds. Fundamental progress.</span>
          </div>

          <div className={`hero-composition${refracted ? " is-refracted" : ""}`}>
            <div className="hero-art">
              <img
                className="dream-world world-one"
                src="/dream-desert.webp"
                alt="A surreal painted desert with a monumental doorway opening onto an ocean, a levitating sun, and an impossible staircase"
                aria-hidden={refracted}
                fetchPriority="high"
                width="1536"
                height="864"
              />
              <img
                className="dream-world world-two"
                src="/dream-sky.webp"
                alt="A painterly dreamscape with an arch in the sky, clouds, and a staircase ascending from an ocean"
                aria-hidden={!refracted}
                width="1536"
                height="864"
              />
            </div>
            <div className="hero-copy">
              <h1>
                Intelligence,
                <br />
                <em>in a new light.</em>
              </h1>
              <div className="hero-introduction">
                <span className="note-heading">Research, from first principles.</span>
                <p>
                  We develop frontier AI models and the systems that make them practical.
                  Fundamental research, from first principles to real machines.
                </p>
                <a href="#research" className="inline-link">
                  Explore our research <Arrow />
                </a>
              </div>
            </div>
            <a
              className="dream-seal"
              href="#junkie"
              aria-label="Discover Junkie, our upcoming model"
            >
              <svg viewBox="0 0 140 140" aria-hidden="true">
                <defs>
                  <path id="seal-circle" d="M70 15a55 55 0 1 1-.1 0" />
                </defs>
                <circle cx="70" cy="70" r="67" fill="none" stroke="currentColor" strokeWidth=".7" />
                <text>
                  <textPath href="#seal-circle" textLength="343">
                    LUCID MACHINES · INDEPENDENT RESEARCH ·{" "}
                  </textPath>
                </text>
              </svg>
              <Symbol className="seal-eye" />
              <span>Discover Junkie</span>
            </a>
            <button
              type="button"
              className="perspective-control"
              aria-pressed={refracted}
              onClick={() => setRefracted(!refracted)}
            >
              <span className="perspective-icon" aria-hidden="true">
                <i />
                <i />
              </span>
              Change view
              <span>{refracted ? "II" : "I"}</span>
            </button>
            <span className="hero-art-caption">Surreal studies, no. {refracted ? "02" : "01"}</span>
          </div>

          <a className="announcement page-gutter" href="#junkie">
            <span className="announcement-label">
              <span className="status-dot" /> In development
            </span>
            <span>Introducing Junkie, our upcoming frontier model.</span>
            <span className="announcement-action">
              Meet Junkie <Arrow />
            </span>
          </a>
        </section>

        <section id="junkie" className="model-section">
          <div className="model-top page-gutter">
            <SectionLabel number="01">The next model</SectionLabel>
            <span className="model-status">
              <i /> Research & development
            </span>
          </div>
          <div className="model-gallery page-gutter">
            <div className="model-content">
              <div className="model-heading">
                <span className="model-preface">Introducing</span>
                <h2>
                  Junkie<span>01</span>
                </h2>
                <p>
                  Our next step in
                  <br />
                  machine intelligence.
                </p>
              </div>
              <div className="model-description">
                <p>
                  A frontier model in development. Informed by our work in efficient architectures,
                  adaptive computation, and the infrastructure beneath intelligence.
                </p>
                <button
                  ref={modalTrigger}
                  className="light-button"
                  type="button"
                  onClick={openBrief}
                >
                  Read the model brief <Arrow />
                </button>
              </div>
            </div>
            <figure className="model-visual">
              <span className="frame-outline" aria-hidden="true" />
              <span className="paper-moon" aria-hidden="true" />
              <div className="model-art-mat">
                <img
                  src="/dream-mind.webp"
                  loading="lazy"
                  width="1024"
                  height="1280"
                  alt="A surreal marble head with floating sections, clouds, and a sun, resting on an impossible staircase"
                />
              </div>
              <figcaption>
                <span>Form & possibility</span>
                <em>Original visual study, II.</em>
              </figcaption>
            </figure>
          </div>
          <div className="model-bottom page-gutter">
            <span>Capability is a model problem.</span>
            <span>
              Making it useful is a systems problem. <span aria-hidden="true">↓</span>
            </span>
          </div>
        </section>

        <section id="systems" className="systems-section section-space page-gutter">
          <SectionLabel number="02">The work beneath the work</SectionLabel>
          <div className="section-heading systems-heading">
            <h2>
              Research is only
              <br />
              <em>
                the beginning<span className="blue-period">.</span>
              </em>
            </h2>
            <p>
              We build the entire path from an idea to a working model. These are the tools and
              systems we’re building along the way.
            </p>
          </div>

          <div className="system-explorer">
            <div className="system-selector">
              <div
                className="system-tabs"
                role="tablist"
                aria-label="Explore our systems"
                aria-orientation="vertical"
              >
                {systems.map((item, index) => (
                  <button
                    key={item.name}
                    ref={(element) => {
                      systemTabs.current[index] = element;
                    }}
                    id={`system-tab-${index}`}
                    type="button"
                    role="tab"
                    aria-selected={activeSystem === index}
                    aria-controls={`system-panel-${index}`}
                    tabIndex={activeSystem === index ? 0 : -1}
                    className={activeSystem === index ? "is-selected" : ""}
                    onClick={() => setActiveSystem(index)}
                    onKeyDown={(event) => {
                      let next = index;
                      if (event.key === "ArrowDown" || event.key === "ArrowRight")
                        next = (index + 1) % systems.length;
                      else if (event.key === "ArrowUp" || event.key === "ArrowLeft")
                        next = (index - 1 + systems.length) % systems.length;
                      else if (event.key === "Home") next = 0;
                      else if (event.key === "End") next = systems.length - 1;
                      else return;
                      event.preventDefault();
                      setActiveSystem(next);
                      systemTabs.current[next]?.focus();
                    }}
                  >
                    <span className="tab-number">0{index + 1}</span>
                    <span>
                      {item.name}
                      <small>{item.category}</small>
                    </span>
                    <Arrow />
                  </button>
                ))}
              </div>
              <div className="system-summary" key={activeSystem}>
                <p>{system.description}</p>
                <a className="inline-link" href={system.link} target="_blank" rel="noreferrer">
                  {system.action} <Arrow />
                </a>
              </div>
            </div>
            {systems.map((item, index) => (
              <div
                key={item.name}
                id={`system-panel-${index}`}
                role="tabpanel"
                aria-labelledby={`system-tab-${index}`}
                hidden={activeSystem !== index}
                tabIndex={0}
                className="system-panel"
              >
                <div className="system-panel-top">
                  <span>Lucid / {item.name}</span>
                  <span>Architecture study 0{index + 1}</span>
                </div>
                <div className="system-drawing">
                  <SystemDrawing type={index} />
                </div>
                <div className="system-features">
                  {item.features.map((feature) => (
                    <span key={feature}>{feature}</span>
                  ))}
                </div>
                <div className="system-command">
                  <span aria-hidden="true">$</span>
                  <code>{item.command}</code>
                  <span>
                    Open source <Arrow />
                  </span>
                </div>
              </div>
            ))}
          </div>
          <div className="systems-footnote">
            <span>Built in the open. Designed to be inspected.</span>
            <a href={repository} target="_blank" rel="noreferrer">
              View the Triune repository <Arrow />
            </a>
          </div>
        </section>

        <section className="evidence-section page-gutter" aria-labelledby="evidence-title">
          <div className="evidence-heading">
            <p className="eyebrow">Research in practice</p>
            <h2 id="evidence-title">
              Ambitious models.
              <br />
              <em>Smaller footprint.</em>
            </h2>
          </div>
          <div className="evidence-measurement">
            <div className="measurement-number">
              1.52<span>B</span>
            </div>
            <p>
              parameters trained within an
              <br />
              <strong>8 GB consumer GPU.</strong>
            </p>
            <a
              href={`${repository}/blob/main/benchmarks/LAYER_STREAMING_REPORT.md`}
              target="_blank"
              rel="noreferrer"
            >
              Read the layer-streaming benchmark <Arrow />
            </a>
            <span className="evidence-caveat">
              Reported Triune configuration. See methodology and hardware details.
            </span>
          </div>
          <div className="evidence-stair" aria-hidden="true">
            <SurrealStudy type={0} />
          </div>
        </section>

        <section id="research" className="research-section section-space page-gutter">
          <SectionLabel number="03">Open questions. Concrete work.</SectionLabel>
          <div className="section-heading research-heading">
            <h2>
              Not just larger.
              <br />
              <em>Fundamentally better.</em>
            </h2>
            <a
              className="inline-link"
              href={`${repository}/tree/main/papers`}
              target="_blank"
              rel="noreferrer"
            >
              Papers & technical reports <Arrow />
            </a>
          </div>
          <div className="research-layout">
            <div className={`research-plate plate-${activeProgram}`} aria-hidden="true">
              <div className="plate-meta">
                <span>Studies in possibility</span>
                <span>No. 0{activeProgram + 1}</span>
              </div>
              <SurrealStudy type={activeProgram} />
              <div className="plate-footer">
                <span>{programs[activeProgram].label}</span>
                <span>0{activeProgram + 1} / 03</span>
              </div>
            </div>
            <div className="research-programs">
              {programs.map((program, index) => (
                <article
                  key={program.number}
                  className={`research-program${activeProgram === index ? " is-expanded" : ""}`}
                >
                  <h3>
                    <button
                      type="button"
                      id={`program-trigger-${index}`}
                      aria-expanded={activeProgram === index}
                      aria-controls={`program-body-${index}`}
                      onClick={() => setActiveProgram(index)}
                    >
                      <span className="program-number">{program.number}</span>
                      <span>{program.title}</span>
                      <span className="program-toggle" aria-hidden="true">
                        <i />
                        <i />
                      </span>
                    </button>
                  </h3>
                  <div
                    id={`program-body-${index}`}
                    role="region"
                    aria-labelledby={`program-trigger-${index}`}
                    hidden={activeProgram !== index}
                    className="program-body"
                  >
                    <span className="eyebrow">{program.label}</span>
                    <p>{program.description}</p>
                    <span className="program-detail">{program.detail}</span>
                    <a className="inline-link" href={program.link} target="_blank" rel="noreferrer">
                      {program.linkText}
                      <Arrow />
                    </a>
                  </div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section id="company" className="company-section section-space page-gutter">
          <SectionLabel number="04">Lucid Machines</SectionLabel>
          <div className="company-layout">
            <div className="company-art" aria-hidden="true">
              <div className="company-portal">
                <img src="/dream-sky.webp" width="1536" height="864" alt="" loading="lazy" />
              </div>
              <Symbol className="company-symbol" />
              <span>Independent by nature.</span>
            </div>
            <div className="company-copy">
              <h2>
                A research company.
                <br />
                <em>A different perspective.</em>
              </h2>
              <p>
                Lucid Machines works across model architecture, optimization, and the systems that
                connect them. We believe progress comes from questioning the whole stack, not just
                scaling one part of it.
              </p>
              <p>
                Our work is grounded in experiments, open implementations, and a straightforward
                question: what would make intelligence genuinely more useful?
              </p>
              <a href={repository} className="inline-link" target="_blank" rel="noreferrer">
                Get to know our work <Arrow />
              </a>
            </div>
          </div>
          <div className="company-principles">
            <span>
              <b>01</b> Independent thinking
            </span>
            <span>
              <b>02</b> Open methods
            </span>
            <span>
              <b>03</b> Systems-level progress
            </span>
          </div>
        </section>
      </main>

      <footer className="site-footer page-gutter">
        <div className="footer-top">
          <div>
            <p className="eyebrow">The work continues</p>
            <h2>
              Stay close to
              <br />
              <em>what comes next.</em>
            </h2>
          </div>
          <a className="footer-follow" href={repository} target="_blank" rel="noreferrer">
            Follow our research{" "}
            <span>
              <Arrow />
            </span>
          </a>
        </div>
        <div className="footer-navigation">
          <Brand footer />
          <nav aria-label="Footer navigation">
            {navItems.map((item) => (
              <a key={item.href} href={`#${item.href}`}>
                {item.text}
              </a>
            ))}
          </nav>
          <a href="#top" className="back-top">
            Back to top <Arrow />
          </a>
        </div>
        <div className="footer-wordmark" aria-hidden="true">
          <span>lucid</span>
          <em>machines</em>
          <Symbol />
        </div>
        <div className="footer-bottom">
          <span>© {new Date().getFullYear()} Lucid Machines</span>
          <span>Independent artificial intelligence research</span>
          <span>Clarity, by design.</span>
        </div>
      </footer>

      <dialog
        className="model-dialog"
        ref={dialog}
        aria-labelledby="model-dialog-title"
        onClick={(event) => {
          if (event.target === event.currentTarget) dialog.current?.close();
        }}
      >
        <div className="dialog-content">
          <div className="dialog-header">
            <Brand onNavigate={() => dialog.current?.close()} />
            <button
              type="button"
              className="dialog-close"
              aria-label="Close model brief"
              onClick={() => dialog.current?.close()}
            >
              <span />
              <span />
            </button>
          </div>
          <p className="eyebrow">Frontier model / In development</p>
          <h2 id="model-dialog-title">
            Junkie<span>Model brief</span>
          </h2>
          <p className="dialog-intro">The next model from Lucid Machines.</p>
          <p>
            Junkie is our upcoming frontier AI model. Its development draws on our work in adaptive
            computation, efficient sequence modeling, and hardware-aware learning.
          </p>
          <dl>
            <div>
              <dt>Current stage</dt>
              <dd>Research & development</dd>
            </div>
            <div>
              <dt>Technical foundation</dt>
              <dd>The Triune research ecosystem</dd>
            </div>
            <div>
              <dt>Release information</dt>
              <dd>To be announced</dd>
            </div>
          </dl>
          <p className="dialog-note">
            Architecture details, evaluation results, and availability will be shared as the
            research progresses.
          </p>
          <a className="primary-button" href={repository} target="_blank" rel="noreferrer">
            Explore the research foundation <Arrow />
          </a>
        </div>
      </dialog>
    </>
  );
}
