<script lang="ts">
  import ModeToggle from "$lib/components/ModeToggle.svelte";
  import { ShieldCheck } from "lucide-svelte";
</script>

<header
  class="sticky top-0 z-50 w-full border-b border-b/40 bg-background/60 backdrop-blur"
>
  <div class="container flex h-14 max-w-screen-2xl items-center">
    <div class="flex items-center gap-2">
      <ShieldCheck class="size-8 text-primary" />
      <div class="mr-4 font-cubano text-xl flex">News Verifier</div>
    </div>
    <div class="flex-1 flex items-end"></div>
    <ModeToggle />
  </div>
</header>
