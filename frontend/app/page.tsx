import { Console } from "@/components/console";
import { EvidenceSection } from "@/components/evidence";
import { Hero } from "@/components/hero";
import { HowItThinks } from "@/components/how-it-thinks";
import { Nav } from "@/components/nav";

export default function Home() {
  return (
    <>
      <Nav />
      <main>
        <Hero />
        <Console />
        <HowItThinks />
        <EvidenceSection />
      </main>
    </>
  );
}
