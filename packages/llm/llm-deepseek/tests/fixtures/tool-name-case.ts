/** Decode replayed responses as a relay that capitalizes returned tool names would deliver them. */
import type { Context } from '@deepseek-ai/cordis'
import type { FinishReason, StreamChunk } from '@deepseek-ai/dsh-llm'
import { translate } from '../../src/translate.ts'

export const name = 'messages-tool-name-case-snapshot'
export const inject = ['llm']

const STOP_REASONS = { 'stop': 'end_turn', 'tool-calls': 'tool_use', 'max-tokens': 'max_tokens' } as const

function stopReason(reason: FinishReason): string {
  if (reason.kind in STOP_REASONS) return STOP_REASONS[reason.kind as keyof typeof STOP_REASONS]
  throw new Error(`Unsupported replayed finish reason ${reason.kind}`)
}

async function* relayEvents(chunks: AsyncIterable<StreamChunk>): AsyncGenerator<Record<string, unknown>> {
  yield { type: 'message_start', message: { usage: {} } }
  let index = 0
  let usage: Record<string, number | undefined> = {}
  for await (const chunk of chunks) {
    if (chunk.type === 'usage') {
      usage = {
        input_tokens: chunk.usage.inputTokens, output_tokens: chunk.usage.outputTokens,
        cache_read_input_tokens: chunk.usage.cacheReadTokens, cache_creation_input_tokens: chunk.usage.cacheWriteTokens,
      }
    } else if (chunk.type === 'block-end') {
      const block = chunk.block
      let content: Record<string, unknown>
      switch (block.type) {
        case 'text': content = { type: 'text', text: block.text }; break
        case 'reasoning': content = { type: 'thinking', thinking: block.text }; break
        case 'tool-call': {
          const capitalized = block.name.charAt(0).toUpperCase() + block.name.slice(1)
          content = { type: 'tool_use', id: block.id, name: capitalized, input: JSON.parse(block.arguments) }
          break
        }
        default: throw new Error(`Unsupported replayed block ${block.type}`)
      }
      yield { type: 'content_block_start', index, content_block: content }
      yield { type: 'content_block_stop', index }
      index++
    } else if (chunk.type === 'finish') {
      yield { type: 'message_delta', delta: { stop_reason: stopReason(chunk.reason) }, usage }
      yield { type: 'message_stop' }
    }
  }
}

/**
 * Route every replayed response through the Messages decoder with capitalized tool names.
 * @param ctx - Snapshot composition context.
 */
export function apply(ctx: Context): void {
  ctx.on('llm/stream', (options, next) => translate(relayEvents(next()), options.model, options.tools?.map(tool => tool.name)))
}
