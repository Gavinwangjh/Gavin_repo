import re
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from loguru import logger


@dataclass
class ArticleSection:
    level: int
    number: str
    title: str
    content: str
    full_text: str
    start_pos: int
    end_pos: int
    parent_section: Optional[str] = None
    section_path: str = ""  # ✅ 新增：结构化路径
    chunk_type: str = "article"  # ✅ 新增：chunk类型


class ArticleChunker:
    def __init__(self, max_chunk_size: int = 2000, min_chunk_size: int = 100):
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size
        
        # 注意：pattern 里我们会把"完整编号格式"保留下来
        self.article_patterns = [
            (r'^(第[一二三四五六七八九十百千万\d]+[章节])\s*(.*?)$', 1, 'chapter'),       # 第一章 / 第一节
            (r'^(第[一二三四五六七八九十百]+条)\s*(.*?)$', 1, 'article'),      # 第十二条
            (r'^([一二三四五六七八九十]+、)\s*(.*?)$', 1, 'chapter'),          # 一、
            (r'^(（[一二三四五六七八九十]+）)\s*(.*?)$', 2, 'section'),        # （一）
            (r'^(\d+\.)\s*(.*?)$', 3, 'item'),                                  # 1.
            (r'^(\d+、)\s*(.*?)$', 3, 'item'),                                  # 1、
            (r'^(\d+(?:\.\d+)+)\s*(.*?)$', 3, 'item'),                          # 1.1 / 1.1.1
            (r'^(（\d+）)\s*(.*?)$', 4, 'subitem'),                             # （1）
            (r'^([A-Z]\.)\s*(.*?)$', 3, 'alpha_item'),                          # A.
            (r'^([a-z]\))\s*(.*?)$', 4, 'alpha_subitem'),                       # a)
        ]
        
        # ✅ 新增：语义分割符（第三阶段）
        self.semantic_separators = [
            "。", "！", "？", "；",
            ".", "!", "?", ";",
            "，", "、", ",",
            "——", "—", "-", ":", "："
        ]
        
        # ✅ 新增：chunk类型映射
        self.chunk_type_names = {
            'article': '条文',
            'chapter': '章节', 
            'section': '节',
            'item': '条目',
            'subitem': '子条目',
            'alpha_item': '字母条目',
            'alpha_subitem': '字母子条目',
            'paragraph': '段落',
            'list': '列表',
            'table': '表格'
        }

        self.atomic_section_terms = [
            "适用范围", "申报条件", "申请条件", "支持对象", "扶持对象", "补贴对象",
            "扶持标准", "补助标准", "奖励标准", "资助标准", "认定标准",
            "申报材料", "申请材料", "办理流程", "申报流程", "审批流程",
            "责任部门", "主管部门", "牵头部门", "职责分工", "监督管理",
            "实施期限", "有效期", "截止时间", "附则", "解释权", "处罚",
            "违规", "考核", "验收", "公示", "备案",
        ]

    # ==================== 第一阶段：保持现有方法不变 ====================
    
    def _detect_article_structure(self, text: str) -> List[ArticleSection]:
        # 保持现有代码完全不变
        lines_raw = text.split('\n')
        lines = [ln.rstrip("\n") for ln in lines_raw]

        sections: List[ArticleSection] = []
        i = 0
        current_position = 0

        while i < len(lines):
            raw_line = lines[i]
            line = raw_line.strip()

            if not line:
                current_position += len(raw_line) + 1
                i += 1
                continue

            matched = False

            for pattern, level, stype in self.article_patterns:  # ✅ 使用stype
                m = re.match(pattern, line)
                if not m:
                    continue

                number = m.group(1).strip()
                title = (m.group(2) or "").strip()

                content_lines = [line]
                start_pos = current_position
                end_pos = current_position + len(raw_line) + 1

                j = i + 1
                pos_cursor = end_pos

                while j < len(lines):
                    nxt_raw = lines[j]
                    nxt = nxt_raw.strip()

                    if not nxt:
                        content_lines.append("")
                        pos_cursor += len(nxt_raw) + 1
                        j += 1
                        continue

                    is_new = False
                    for npat, nlevel, _ in self.article_patterns:
                        if re.match(npat, nxt):
                            if nlevel <= level:
                                is_new = True
                            break
                    if is_new:
                        break

                    content_lines.append(nxt)
                    pos_cursor += len(nxt_raw) + 1
                    j += 1

                end_pos = pos_cursor
                full_text = "\n".join([x for x in content_lines if x is not None]).strip()
                content = "\n".join([x for x in content_lines[1:] if x is not None]).strip()

                # ✅ 新增：设置chunk_type
                sections.append(ArticleSection(
                    level=level,
                    number=number,
                    title=title,
                    content=content,
                    full_text=full_text,
                    start_pos=start_pos,
                    end_pos=end_pos,
                    chunk_type=stype  # ✅ 新增
                ))

                matched = True
                i = j
                current_position = end_pos
                break

            if not matched:
                current_position += len(raw_line) + 1
                i += 1

        return sections

    def _assign_parent_sections(self, sections: List[ArticleSection]) -> List[ArticleSection]:
        # 保持现有代码
        for idx, sec in enumerate(sections):
            for j in range(idx - 1, -1, -1):
                if sections[j].level < sec.level:
                    sec.parent_section = sections[j].number
                    break
        return sections
    
    # ==================== 第二阶段：增强现有方法 ====================
    
    def _build_section_path(self, sections: List[ArticleSection]) -> List[ArticleSection]:
        """构建结构化路径（第三阶段新增）"""
        for sec in sections:
            path_parts = []
            
            # 如果有父级，添加到路径
            if sec.parent_section:
                # 找到父级的完整路径
                for parent_sec in sections:
                    if parent_sec.number == sec.parent_section:
                        if parent_sec.section_path:
                            path_parts.append(parent_sec.section_path)
                        else:
                            path_parts.append(parent_sec.number)
                        break
            
            # 添加当前编号
            path_parts.append(sec.number)
            
            # 构建路径字符串
            sec.section_path = " > ".join(path_parts)
        
        return sections
    
    def _merge_small_sections_semantic(self, sections: List[ArticleSection]) -> List[ArticleSection]:
        """语义感知的小段落合并（第三阶段增强）"""
        if not sections:
            return sections
        
        merged: List[ArticleSection] = []
        i = 0
        
        while i < len(sections):
            cur = sections[i]
            
            # 如果当前段落足够大，直接保留
            if len(cur.full_text) >= self.min_chunk_size:
                merged.append(cur)
                i += 1
                continue
            
            # 尝试与后续同级段落合并
            merge_buffer = [cur.full_text]
            merge_numbers = [cur.number]
            merge_start = cur.start_pos
            merge_end = cur.end_pos
            j = i + 1
            
            # 只合并同级且类型相同的段落
            while j < len(sections):
                nxt = sections[j]
                
                # 检查是否同级同类型
                if nxt.level != cur.level or nxt.chunk_type != cur.chunk_type:
                    break
                
                # 检查合并后是否超过最小尺寸
                if len("\n\n".join(merge_buffer)) + len(nxt.full_text) > self.min_chunk_size:
                    break
                
                merge_buffer.append(nxt.full_text)
                merge_numbers.append(nxt.number)
                merge_end = nxt.end_pos
                j += 1
            
            # 构建合并后的段落
            merged_text = "\n\n".join(merge_buffer).strip()
            merged_number = " + ".join(merge_numbers)
            
            merged_section = ArticleSection(
                level=cur.level,
                number=merged_number,
                title=cur.title,
                content=merged_text,
                full_text=merged_text,
                start_pos=merge_start,
                end_pos=merge_end,
                parent_section=cur.parent_section,
                chunk_type=cur.chunk_type,
                section_path=cur.section_path  # 保持原有路径
            )
            
            merged.append(merged_section)
            i = j  # 跳到下一个未处理的段落
        
        return merged

    def _should_preserve_atomic_sections(self, text: str, sections: List[ArticleSection]) -> bool:
        """Keep law/policy headings as retrieval anchors instead of merging short sections."""
        if not sections:
            return False

        body = text or ""
        if any(term in body for term in self.atomic_section_terms):
            return True

        legal_article_count = sum(
            1 for sec in sections
            if sec.chunk_type == "article" or re.match(r"^第.+条$", sec.number or "")
        )
        policy_heading_count = sum(
            1 for sec in sections
            if sec.chunk_type in {"chapter", "section", "item"} and len(sec.full_text) >= 10
        )

        return legal_article_count >= 2 or policy_heading_count >= 3
    
    # ==================== 第三阶段：新增核心方法 ====================
    
    def _recursive_split_semantic(self, text: str, depth: int = 0) -> List[Tuple[str, str, int]]:
        """
        语义递归切割（第三阶段核心）
        返回：(文本片段, 分割类型, 深度)
        """
        text = (text or "").strip()
        if not text:
            return []
        
        # 基础条件：文本足够小或达到最大深度
        if len(text) <= self.max_chunk_size or depth >= 3:
            split_type = "proper" if len(text) <= self.max_chunk_size else f"hard_depth_{depth}"
            return [(text, split_type, depth)]
        
        # 按优先级尝试不同分割方式
        split_results = []
        
        # 1. 尝试空行分割（段落）
        paragraphs = re.split(r'\n{2,}', text)
        if len(paragraphs) > 1:
            for para in paragraphs:
                if para.strip():
                    if len(para) <= self.max_chunk_size:
                        split_results.append((para.strip(), "paragraph", depth))
                    else:
                        # 递归处理长段落
                        split_results.extend(self._recursive_split_semantic(para, depth + 1))
            return split_results
        
        # 2. 尝试句子分割
        sentence_endings = r'[。！？；.!?;]'
        sentences = re.split(f'({sentence_endings}+)', text)
        
        # 重组句子（保留标点）
        combined = []
        current = ""
        for i in range(0, len(sentences), 2):
            if i < len(sentences):
                sentence = sentences[i]
                if i + 1 < len(sentences):
                    sentence += sentences[i + 1]
                
                if current and len(current) + len(sentence) > self.max_chunk_size:
                    combined.append(current)
                    current = sentence
                else:
                    current += sentence
        
        if current:
            combined.append(current)
        
        if len(combined) > 1:
            for sent in combined:
                if sent.strip():
                    if len(sent) <= self.max_chunk_size:
                        split_results.append((sent.strip(), "sentence", depth))
                    else:
                        split_results.extend(self._recursive_split_semantic(sent, depth + 1))
            return split_results
        
        # 3. 尝试语义分割符（逗号等）
        for sep in self.semantic_separators:
            if sep in text:
                parts = text.split(sep)
                if len(parts) > 1:
                    # 重新添加分隔符
                    semantic_parts = []
                    for part in parts[:-1]:
                        semantic_parts.append(part + sep)
                    semantic_parts.append(parts[-1])
                    
                    for part in semantic_parts:
                        if part.strip():
                            if len(part) <= self.max_chunk_size:
                                split_results.append((part.strip(), f"semantic_{sep}", depth))
                            else:
                                split_results.extend(self._recursive_split_semantic(part, depth + 1))
                    return split_results
        
        # 4. 硬切（字符级）
        for i in range(0, len(text), self.max_chunk_size):
            chunk = text[i:i + self.max_chunk_size]
            if chunk.strip():
                split_results.append((chunk.strip(), f"hard_{depth}", depth))
        
        return split_results
    
    def _split_large_sections_enhanced(self, sections: List[ArticleSection]) -> List[ArticleSection]:
        """增强的大段落分割（第三阶段）"""
        result: List[ArticleSection] = []
        
        for sec in sections:
            if len(sec.full_text) <= self.max_chunk_size:
                result.append(sec)
                continue
            
            # 使用语义递归切割
            split_parts = self._recursive_split_semantic(sec.full_text)
            
            for idx, (chunk_text, split_type, depth) in enumerate(split_parts, 1):
                # 检查句子是否被截断
                has_truncation = self._check_truncation(chunk_text)
                
                # 构建新section
                new_section = ArticleSection(
                    level=sec.level,
                    number=f"{sec.number}#{idx}",
                    title=sec.title,
                    content=chunk_text,
                    full_text=chunk_text,
                    start_pos=sec.start_pos,  # 近似位置
                    end_pos=sec.end_pos,      # 近似位置
                    parent_section=sec.parent_section,
                    chunk_type=f"{sec.chunk_type}_{split_type}",
                    section_path=sec.section_path
                )
                result.append(new_section)
                
                logger.debug(f"分割段落: {sec.number} -> {new_section.number}, "
                           f"类型: {split_type}, 深度: {depth}, "
                           f"截断: {has_truncation}, 长度: {len(chunk_text)}")
        
        return result
    
    def _check_truncation(self, text: str) -> bool:
        """检查句子是否被截断"""
        if not text:
            return False
        
        # 句子结束符
        sentence_endings = {'。', '！', '？', '；', '.', '!', '?', ';'}
        
        # 常见的中文结尾词（可能被截断）
        truncation_patterns = [
            r'[的之地得]$',  # 定语结尾
            r'[和与及以及]$',  # 连接词
            r'[因为所以然而但是]$',  # 连词
            r'[在从向对]$',  # 介词
        ]
        
        last_char = text[-1]
        
        # 如果最后字符不是句子结束符
        if last_char not in sentence_endings:
            # 检查是否符合截断模式
            for pattern in truncation_patterns:
                if re.search(pattern, text[-2:] if len(text) >= 2 else text):
                    return True
            
            # 长度足够但不是句子结束，可能被截断
            if len(text) > 50:
                return True
        
        return False
    
    # ==================== 整合方法 ====================
    
    def chunk_by_articles_enhanced(self, text: str) -> List[Dict[str, Any]]:
        """
        增强的条文分块（第三阶段完整实现）
        """
        try:
            # 1. 检测条文结构
            sections = self._detect_article_structure(text)
            logger.info(f"📋 检测到 {len(sections)} 个条文段落")
            
            if not sections:
                logger.warning("未检测到条文结构，使用增强的回退分块")
                return self._fallback_chunking_enhanced(text)
            
            # 2. 构建父子关系和路径
            sections = self._assign_parent_sections(sections)
            sections = self._build_section_path(sections)
            
            # 3. 语义感知的合并与分割
            if self._should_preserve_atomic_sections(text, sections):
                logger.info("Keeping atomic law/policy sections for precise RAG retrieval")
            else:
                sections = self._merge_small_sections_semantic(sections)
            sections = self._split_large_sections_enhanced(sections)
            
            # 4. 生成增强的chunk元数据
            chunks = []
            for i, sec in enumerate(sections):
                # 计算统计信息
                word_count = len(sec.full_text.split())
                sentence_count = len(re.split(r'[。！？；.!?;]+', sec.full_text))
                has_truncation = self._check_truncation(sec.full_text)
                
                # 生成chunk哈希
                chunk_hash = hashlib.md5(sec.full_text.encode()).hexdigest()[:16]
                
                # 构建完整元数据
                chunks.append({
                    "content": sec.full_text,
                    "chunk_index": i,
                    "article_number": sec.number,
                    "article_title": sec.title,
                    "article_level": sec.level,
                    "parent_article": sec.parent_section,
                    "section_path": sec.section_path,  # ✅ 第三阶段新增
                    "chunk_type": sec.chunk_type,      # ✅ 第三阶段新增
                    "chunk_type_name": self.chunk_type_names.get(sec.chunk_type, "未知"),
                    "metadata": {
                        "start_pos": sec.start_pos,
                        "end_pos": sec.end_pos,
                        "content_length": len(sec.full_text),
                        "word_count": word_count,
                        "sentence_count": sentence_count,
                        "has_truncation": has_truncation,
                        "chunk_hash": chunk_hash,
                        "depth": sec.chunk_type.count('_') + 1 if '_' in sec.chunk_type else 1
                    }
                })
            
            logger.info(f"✅ 增强条文分块完成，生成 {len(chunks)} 个块")
            
            # 输出统计信息
            self._print_chunk_statistics(chunks)
            
            return chunks
            
        except Exception as e:
            logger.error(f"❌ 增强条文分块失败: {e}")
            return self._fallback_chunking_enhanced(text)
    
    def _fallback_chunking_enhanced(self, text: str) -> List[Dict[str, Any]]:
        """增强的回退分块"""
        text = (text or "").strip()
        if not text:
            return []
        
        # 使用语义递归切割
        split_parts = self._recursive_split_semantic(text)
        
        chunks = []
        for i, (chunk_text, split_type, depth) in enumerate(split_parts):
            has_truncation = self._check_truncation(chunk_text)
            chunk_hash = hashlib.md5(chunk_text.encode()).hexdigest()[:16]
            
            chunks.append({
                "content": chunk_text,
                "chunk_index": i,
                "chunk_type": f"fallback_{split_type}",
                "chunk_type_name": "回退分块",
                "metadata": {
                    "content_length": len(chunk_text),
                    "split_type": split_type,
                    "depth": depth,
                    "has_truncation": has_truncation,
                    "chunk_hash": chunk_hash
                }
            })
        
        logger.info(f"🔄 增强回退分块完成，生成 {len(chunks)} 个块")
        return chunks
    
    def _print_chunk_statistics(self, chunks: List[Dict[str, Any]]):
        """输出分块统计信息"""
        if not chunks:
            return
        
        total_chunks = len(chunks)
        total_length = sum(len(c["content"]) for c in chunks)
        avg_length = total_length / total_chunks
        
        # 类型统计
        type_stats = {}
        for chunk in chunks:
            chunk_type = chunk.get("chunk_type", "unknown")
            type_stats[chunk_type] = type_stats.get(chunk_type, 0) + 1
        
        # 截断统计
        truncated_count = sum(1 for c in chunks 
                            if c.get("metadata", {}).get("has_truncation", False))
        
        logger.info(f"📊 分块统计:")
        logger.info(f"  • 总块数: {total_chunks}")
        logger.info(f"  • 平均长度: {avg_length:.0f} 字符")
        logger.info(f"  • 截断块数: {truncated_count} ({truncated_count/total_chunks*100:.1f}%)")
        logger.info(f"  • 类型分布:")
        for chunk_type, count in type_stats.items():
            name = self.chunk_type_names.get(chunk_type.split('_')[0], chunk_type)
            logger.info(f"    - {name}: {count} 个")
    
    # ==================== 保持向后兼容 ====================
    
    def chunk_by_articles(self, text: str) -> List[Dict[str, Any]]:
        """保持向后兼容的原有方法"""
        # 可以调用增强版本，或保持原有逻辑
        return self.chunk_by_articles_enhanced(text)
    
    def is_article_document(self, text: str) -> bool:
        """保持原有方法不变"""
        article_count = 0
        for pattern, _, _ in self.article_patterns:
            article_count += len(re.findall(pattern, text or "", re.MULTILINE))

        lines = (text or "").split("\n")
        non_empty = [l for l in lines if l.strip()]
        if not non_empty:
            return False

        ratio = article_count / len(non_empty)
        policy_terms = [
            "适用范围", "申报条件", "申请条件", "支持对象", "扶持标准",
            "申报材料", "办理流程", "责任部门", "监督管理", "实施期限",
            "附则", "解释权",
        ]
        has_policy_terms = any(term in (text or "") for term in policy_terms)
        return (article_count >= 3 and ratio > 0.03) or (has_policy_terms and article_count >= 2)
