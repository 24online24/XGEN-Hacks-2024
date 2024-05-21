<script lang="ts">
  export let name: string; // name of model
  export let value: number; // confidence
  export let description: string; // description of the model

  // prediction color red if values under 0.33, yellow if under 0.66, green if above 0.66
  let color =
    value < 0.33
      ? "text-red-500"
      : value < 0.66
        ? "text-yellow-500"
        : "text-green-500";

  // a bit more fun with the prediction status
  let predictionMessage;
  value < 0.1
    ? (predictionMessage = "Not so sure")
    : value < 0.5
      ? (predictionMessage = "Maybe")
      : value < 0.9
        ? (predictionMessage = "Pretty sure")
        : (predictionMessage = "Very sure");

  $: confidence = (value * 100).toFixed(2);
</script>

<!-- Animate the height card on hover to show nicely the rest of details -->
<div
  class="bg-card border border-border rounded-lg p-4 mt-4 group cursor-pointer"
>
  <div class="flex items-center justify-between">
    <div class="flex items-center">
      <span class="text-sm capitalize font-medium">{name}</span>
    </div>
    <span class={"text-sm font-medium text-muted-foreground"}
      >{predictionMessage}</span
    >
    <span class={color + " text-sm font-medium"}>{value * 100}% Confidence</span
    >
  </div>
  <div
    class="hidden group-hover:block transition-[height] duration-1000 ease-in-out"
  >
    <p
      class="mt-2 h-0 group-hover:h-auto transition-[height] duration-1000 ease-in-out"
    >
      {description}
    </p>
  </div>
</div>
