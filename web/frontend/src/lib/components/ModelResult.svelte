<script lang="ts">
  export let name: string; // name of model
  export let value: number; // confidence
  export let description: string; // description of the model
  export let accuracy: number; // accuracy of the model

  // prediction color red if values under 0.33, yellow if under 0.66, green if above 0.66
  let color =
    value < 33
      ? "text-red-500"
      : value < 66
        ? "text-yellow-500"
        : "text-green-500";

  // a bit more fun with the prediction status
  let predictionMessage;
  value < 10
    ? (predictionMessage = "Not so sure")
    : value < 50
      ? (predictionMessage = "Maybe")
      : value < 80
        ? (predictionMessage = "Pretty sure")
        : (predictionMessage = "Very sure");

  $: confidence = value.toFixed(2);
</script>

<!-- Animate the height card on hover to show nicely the rest of details -->
<div
  class="bg-card border border-border rounded-lg p-4 mt-4 group cursor-pointer"
>
  <div class="flex items-center justify-between">
    <div class="flex items-center">
      <span class="text-sm capitalize font-medium">{name}</span>
    </div>
    <span class={"text-sm flex-1 text-center font-medium text-muted-foreground"}
      >{predictionMessage}</span
    >
    <span class={color + " text-sm font-medium"}>{value}% Confidence</span>
  </div>
  <div
    class="hidden group-hover:block transition-[height] duration-1000 ease-in-out"
  >
    <p
      class="mt-2 text-xs h-0 group-hover:h-auto transition-[height] duration-1000 ease-in-out"
    >
      {description}
      {#if accuracy > 0}
        <br />
        <span class="text-xs text-muted-foreground">
          Training accuracy: {accuracy}%
        </span>
      {/if}
    </p>
  </div>
</div>
